"""Guarded reads plus deterministic exact publication for FalkorDB.

All generated identities/timestamps and normalization are resolved before the
write while WATCH protects the graph key. The exact record changes, next HEAD,
and immutable receipt are published in ONE GRAPH.QUERY. No after-write capture.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

from potpie_context_engine.core.graph_journal import (
    JournalCapability,
    JournalError,
    JournalLimits,
    JournalRecord,
    JournalState,
    ResourceGuard,
    decode_journal,
    journal_hash,
    journal_json,
)
from potpie_context_engine.core.journal_context import current_journal_context
from potpie_context_engine.core.journal_inverse import (
    plan_inverse,
    protocols_enabled_for,
    validate_state,
)
from potpie_context_engine.core.reconciliation import MutationResult, MutationSummary
from redis.exceptions import ResponseError, WatchError

from ._mutation_execution import (
    MutationExecutionReuseError,
    complete_execution,
)
from .falkordb_connection import transaction_client
from .falkordb_journal_publication import publish_query as _publish_query

_STATE = "PotpieRevision"
_RECEIPT = "PotpieMutationReceipt"


def _read(graph, pipe, query, params):
    from .falkordb_atomic import _rows

    # A graph key that does not exist yet answers every read with "empty key",
    # whether the read goes straight to the graph or through a watched pipeline.
    try:
        if pipe is None:
            return graph.ro_query(query, params=params).result_set
        return _rows(
            graph,
            pipe.execute_command(
                "GRAPH.RO_QUERY",
                graph.name,
                graph._build_params_header(params) + query,
                "--compact",
            ),
        )
    except ResponseError as exc:
        if "empty key" in str(exc).lower():
            return []
        raise


def _state(graph, pot_id, pipe=None):
    rows = _read(
        graph,
        pipe,
        f"MATCH (v:{_STATE} {{pot_id:$pot}}) RETURN v.journal",
        {"pot": pot_id},
    )
    return decode_journal(json.loads(rows[0][0])) if rows and rows[0][0] else None


def _records(graph, pot_id, pipe=None, keys=None):
    """Read touched closure, including singleton siblings and invalidation targets."""
    params = {"pot": pot_id, "keys": list(keys) if keys is not None else None}
    edge_rows = _read(
        graph,
        pipe,
        "MATCH (a:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(b:Entity {group_id:$pot}) "
        "WHERE $keys IS NULL OR a.entity_key IN $keys OR b.entity_key IN $keys OR r.claim_key IN $keys OR coalesce(r.record_id,r.uuid) IN $keys "
        "RETURN a.entity_key,b.entity_key,properties(r)",
        params,
    )
    if keys is not None:
        params["keys"] = sorted(set(keys) | {k for row in edge_rows for k in row[:2]})
    node_rows = _read(
        graph,
        pipe,
        "MATCH (e:Entity {group_id:$pot}) WHERE $keys IS NULL OR e.entity_key IN $keys "
        "RETURN e.entity_key,labels(e),properties(e)",
        params,
    )
    records, entity_ids = {}, {}
    for key, labels, props in node_rows:
        record_id = props.get("record_id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in records
            or key in entity_ids
        ):
            raise JournalError("missing or ambiguous entity record identity")
        entity_ids[key] = record_id
        records[record_id] = JournalRecord(
            record_id,
            pot_id,
            key,
            "entity",
            {
                "active": not props.get("retired", False),
                "labels": tuple(sorted(labels)),
                "properties": {k: v for k, v in props.items() if k != "retired"},
            },
        )
    for source, target, props in edge_rows:
        record_id = props.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in records:
            raise JournalError("missing or ambiguous claim record identity")
        key = (
            props.get("claim_key")
            or f"{source}:{props.get('name')}:{target}:{props.get('source_ref')}"
        )
        records[record_id] = JournalRecord(
            record_id,
            pot_id,
            key,
            "claim",
            {
                "active": not props.get("retired", False),
                "predicate": props["name"],
                "subject_key": source,
                "object_key": target,
                "subject_record_id": entity_ids[source],
                "object_record_id": entity_ids[target],
                "properties": {k: v for k, v in props.items() if k != "retired"},
            },
        )
    return records


def _props(record):
    return deepcopy(dict(record.fields["properties"]))


def _replace_props(record, props, **fields):
    return replace(record, fields={**record.fields, "properties": props, **fields})


def _effect_plan(before, plan, params, definition):
    """Evaluate the existing compiler's normalized parameters exactly once."""
    after = deepcopy(before)
    entities = {r.logical_key: r for r in after.values() if r.kind == "entity"}
    counts = MutationSummary()
    for i, item in enumerate(plan.entity_upserts):
        old = entities.get(item.entity_key)
        if item.preserve_labels and old is None:
            raise JournalError("entity projection target is missing or ambiguous")
        props = (
            _props(old)
            if old
            else {
                "uuid": uuid.uuid4().hex,
                "created_at": int(datetime.now(timezone.utc).timestamp() * 1000),
            }
        )
        record_id = old.record_id if old else props["uuid"]
        raw = deepcopy(params[f"ep{i}"])
        active = old.fields.get("active", True) if old else True
        if "retired" in raw:
            retired = raw["retired"]
            active = retired is None or retired is False
        if params.get("journal_generated_event"):
            raw["execution_source_event_id"] = raw.pop("prov_source_event_id", None)
            raw.pop("provenance_source_event", None)
        if params.get("journal_generated_valid_from"):
            raw["execution_valid_from"] = raw.pop("prov_valid_from", None)
        for key, value in raw.items():
            if value is None:
                props.pop(key, None)
            else:
                props[key] = value
        for prop, param in (
            ("name", f"ean{i}"),
            ("summary", f"eas{i}"),
            ("description", f"ead{i}"),
        ):
            if item.preserve_labels:
                continue
            value = params[param]
            if value:
                props[prop] = value
            elif not props.get(prop):
                props[prop] = item.entity_key
        props.update(
            entity_key=item.entity_key, group_id=params["pot"], record_id=record_id
        )
        props.pop("retired", None)
        labels = set(old.fields["labels"] if old else ("Entity",))
        from .cypher import coherent_entity_labels

        if not item.preserve_labels:
            labels.difference_update(definition.entity_types)
            labels.update(
                coherent_entity_labels(
                    item.entity_key, item.labels, entity_types=definition.entity_types
                )
            )
        record = JournalRecord(
            record_id,
            params["pot"],
            item.entity_key,
            "entity",
            {"active": active, "labels": tuple(sorted(labels)), "properties": props},
        )
        after[record_id] = entities[item.entity_key] = record
        counts.entity_upserts_applied += 1

    # The watched native resolution is also the journal's exact after-state.
    # No semantic selectors or lifecycle choices are repeated during capture.
    for write in params["claim_writes"]:
        props = deepcopy(write["props"])
        record_id = write["record_id"]
        source, target = write["source"], write["target"]
        a, b = entities[source], entities[target]
        after[record_id] = JournalRecord(
            record_id,
            params["pot"],
            props.get("claim_key")
            or f"{source}:{props['name']}:{target}:{props.get('source_ref')}",
            "claim",
            {
                "active": not props.get("retired", False),
                "predicate": props["name"],
                "subject_key": source,
                "object_key": target,
                "subject_record_id": a.record_id,
                "object_record_id": b.record_id,
                "properties": {
                    key: value
                    for key, value in props.items()
                    if value is not None and key != "retired"
                },
            },
        )
    counts.edge_upserts_applied = params["claim_edges"]
    counts.edge_deletes_applied = params["claim_deletes"]
    counts.invalidations_applied = params["claim_invalidations"]
    return after, counts


class FalkorJournal:
    def __init__(self, graph, definition, *, profile="falkordb", resource_exists=None):
        self.graph, self.definition, self.profile = graph, definition, profile
        self.resource_exists = resource_exists

    def journal_state(self, pot_id):
        return _state(self.graph, pot_id)

    def journal_capability(self, pot_id):
        state = self.journal_state(pot_id)
        supported = self.profile == "falkordb"
        return JournalCapability(
            supported,
            bool(supported and state and state.rollback_enabled),
            supported,
            None if supported else "FalkorDBLite awaits its independent live gate",
        )

    def _execute(self, pipe, query, params):
        from .falkordb_atomic import _rows
        from .falkordb_revision import read_revision

        read_revision(self.graph, params["pot"], pipe)
        pipe.multi()
        pipe.execute_command(
            "GRAPH.QUERY",
            self.graph.name,
            self.graph._build_params_header(params) + query,
            "--compact",
        )
        try:
            rows = _rows(self.graph, pipe.execute()[0])
        except WatchError as exc:
            raise JournalError(
                "native journal publication guard changed", code="preview_stale"
            ) from exc
        if not rows:
            raise JournalError("native journal publication guard changed")

    def activate(self, *, pot_id, rollback_enabled=False, limits=None):
        if self.profile != "falkordb":
            raise JournalError(
                "this backend's journal capability has not passed its live gate"
            )
        from .falkordb_atomic import _ensure_state

        _ensure_state(self.graph, pot_id)
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            existing = _state(self.graph, pot_id, pipe)
            if existing:
                return existing
            state = JournalState(
                pot_id,
                uuid.uuid4().hex,
                rollback_enabled=rollback_enabled,
                limits=limits or JournalLimits(),
            )
            nodes = _read(
                self.graph,
                pipe,
                "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key,e.uuid,e.record_id",
                {"pot": pot_id},
            )
            edges = _read(
                self.graph,
                pipe,
                "MATCH (a:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(b:Entity {group_id:$pot}) RETURN r.uuid,r.record_id",
                {"pot": pot_id},
            )
            ids, keys = set(), set()
            for key, uid, record_id in nodes:
                if (
                    not isinstance(key, str)
                    or not key
                    or any(
                        value is not None and value != "" and not isinstance(value, str)
                        for value in (uid, record_id)
                    )
                ):
                    raise JournalError("invalid native entity migration identity")
                if key in keys or (record_id or uid) in ids:
                    raise JournalError("ambiguous entity migration")
                keys.add(key)
                if record_id or uid:
                    ids.add(record_id or uid)
            for uid, record_id in edges:
                if any(
                    value is not None and value != "" and not isinstance(value, str)
                    for value in (uid, record_id)
                ):
                    raise JournalError("invalid native claim migration identity")
                if (record_id or uid) and (record_id or uid) in ids:
                    raise JournalError("ambiguous relationship migration")
                if record_id or uid:
                    ids.add(record_id or uid)
            # Assign missing UUIDs and persistent record IDs in the SAME
            # guarded activation command, not backend-internal relationship IDs.
            query = (
                f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.journal IS NULL "
                "CALL { WITH v OPTIONAL MATCH (e:Entity {group_id:$pot}) "
                "FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END | SET e.record_id=coalesce(CASE WHEN e.record_id='' THEN null ELSE e.record_id END,CASE WHEN e.uuid='' THEN null ELSE e.uuid END,randomUUID())) RETURN count(e) AS nodes } "
                "CALL { WITH v OPTIONAL MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
                "FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END | SET r.record_id=coalesce(CASE WHEN r.record_id='' THEN null ELSE r.record_id END,CASE WHEN r.uuid='' THEN null ELSE r.uuid END,randomUUID())) RETURN count(r) AS edges } "
                "SET v.journal=$journal RETURN v.version"
            )
            self._execute(pipe, query, {"pot": pot_id, "journal": journal_json(state)})
            return state

    def set_rollback_enabled(self, *, pot_id, enabled):
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            state = _state(self.graph, pot_id, pipe)
            if state is None:
                raise JournalError("journal is not active")
            updated = replace(state, rollback_enabled=enabled)
            self._execute(
                pipe,
                f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.journal=$old "
                "SET v.journal=$next RETURN v.version",
                {
                    "pot": pot_id,
                    "old": journal_json(state),
                    "next": journal_json(updated),
                },
            )
            return updated

    def get_receipt(self, *, pot_id, commit_id):
        rows = _read(
            self.graph,
            None,
            f"MATCH (r:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) RETURN r.journal",
            {"pot": pot_id, "mid": commit_id},
        )
        if len(rows) > 1:
            raise JournalError("ambiguous native receipt")
        return decode_journal(json.loads(rows[0][0])) if rows and rows[0][0] else None

    def read_receipts(self, *, pot_id, generation, after_sequence, limit=100):
        if not 1 <= limit <= 200:
            raise JournalError("receipt page size must be between 1 and 200")
        rows = _read(
            self.graph,
            None,
            f"MATCH (r:{_RECEIPT} {{pot_id:$pot,journal_generation:$generation}}) "
            "WHERE r.sequence>$after RETURN r.journal ORDER BY r.sequence LIMIT $limit",
            {
                "pot": pot_id,
                "generation": generation,
                "after": after_sequence,
                "limit": limit,
            },
        )
        return tuple(decode_journal(json.loads(row[0])) for row in rows)

    def _target_receipts(self, state, target_id, mode):
        target = self.get_receipt(pot_id=state.pot_id, commit_id=target_id)
        if target is None:
            return ()
        if mode == "revert":
            return (target,)
        receipts, sequence = [target], target.sequence
        while sequence < state.sequence:
            page = self.read_receipts(
                pot_id=state.pot_id,
                generation=state.generation,
                after_sequence=sequence,
                limit=200,
            )
            if not page:
                break
            receipts.extend(page)
            sequence = page[-1].sequence
        return receipts

    def _receipt(self, state, before, after, **kwargs):
        # One shared capture policy across native profiles.
        from potpie_context_engine.core.journal_capture import capture_receipt

        return capture_receipt(state, before, after, **kwargs)

    def apply(
        self,
        plan,
        *,
        pipe,
        state,
        params,
        fingerprint,
        actor,
        provenance_context,
        receipt_batch,
    ):
        context = current_journal_context()
        if context.resource_operation_id and state.resource_guard is None:
            raise JournalError("resource operation no longer owns an active guard")
        if state.resource_guard and (
            context.resource_operation_id,
            context.resource_owner,
        ) != (state.resource_guard.operation_id, state.resource_guard.owner):
            raise JournalError("resource operation owns the pot write guard")
        keys = {item.entity_key for item in plan.entity_upserts}
        for write in params["claim_writes"]:
            keys.update((write["record_id"], write["source"], write["target"]))
        before = _records(self.graph, state.pot_id, pipe, keys=keys)
        after, summary = _effect_plan(before, plan, params, self.definition)
        receipt = self._receipt(
            state,
            before,
            after,
            commit_id=params["mid"],
            fingerprint=fingerprint,
            actor=actor,
            message=plan.summary,
        )
        result = complete_execution(
            MutationResult(
                True, receipt.commit_id, summary, downgrades=plan.ontology_downgrades
            ),
            receipt_batch,
            pot_id=state.pot_id,
            version_before=params["expected"],
            provenance=provenance_context,
            applied_at=datetime.fromisoformat(params["now"]),
            record_ids=(
                key
                for key, value in after.items()
                if value.kind == "claim" and before.get(key) != value
            ),
        )
        query, values = _publish_query(
            before,
            after,
            state=state,
            receipt=receipt,
            result=summary,
            execution=result,
        )
        try:
            self._execute(pipe, query, values)
        except JournalError as exc:
            # The batch executor re-reads its durable receipt after contention.
            # Public restore/resource operations retain their stale-preview error.
            if isinstance(exc.__cause__, WatchError):
                raise exc.__cause__
            raise
        return result

    def plan_restore(self, *, pot_id, target_commit_id, expected_head, mode="revert"):
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            state = _state(self.graph, pot_id, pipe)
            if state is None:
                raise JournalError("journal is not active")
            return self._plan(
                state,
                _records(self.graph, pot_id, pipe),
                target_commit_id,
                expected_head,
                mode,
            )

    def _plan(self, state, records, target, head, mode):
        return plan_inverse(
            state=state,
            current=records,
            receipts=self._target_receipts(state, target, mode),
            target_commit_id=target,
            expected_head=head,
            mode=mode,
            validator=lambda body: validate_state(
                body,
                pot_id=state.pot_id,
                definition=self.definition,
                protocols_enabled=protocols_enabled_for(self.definition),
                resource_exists=(lambda ref: self.resource_exists(state.pot_id, ref))
                if self.resource_exists
                else None,
            ),
        )

    def apply_restore(self, plan, *, mutation_id, actor):
        fingerprint = journal_hash(
            {
                "inverse": plan.inverse_hash,
                "actor": actor,
                "generation": plan.journal_generation,
                "head": plan.expected_head,
                "mode": plan.mode,
                "target": plan.target_commit_id,
                "access": plan.required_access,
                "resource_generation": plan.resource_generation,
            }
        )
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            prior = self.get_receipt(pot_id=plan.pot_id, commit_id=mutation_id)
            if prior:
                if prior.fingerprint != fingerprint:
                    raise MutationExecutionReuseError(
                        "commit ID reused with different restore"
                    )
                return prior
            legacy = _read(
                self.graph,
                pipe,
                f"MATCH (r:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) RETURN r.fingerprint",
                {"pot": plan.pot_id, "mid": mutation_id},
            )
            if legacy:
                raise JournalError("legacy mutation ID was reused for a restore")
            state = _state(self.graph, plan.pot_id, pipe)
            if state is None:
                raise JournalError("journal is not active")
            before = _records(self.graph, plan.pot_id, pipe)
            if (
                self._plan(
                    state, before, plan.target_commit_id, plan.expected_head, plan.mode
                )
                != plan
            ):
                raise JournalError("restore plan or resource generation changed")
            after = deepcopy(before)
            for patch in plan.records:
                from potpie_context_engine.core.journal_fields import clear_derived

                restored = clear_derived(patch.after)
                props = _props(restored)
                props.update(
                    mutation_id=mutation_id,
                    prov_mutation_id=mutation_id,
                    prov_graph_updated_at=datetime.now(timezone.utc).isoformat(),
                )
                after[patch.after.record_id] = _replace_props(restored, props)
            receipt = self._receipt(
                state,
                before,
                after,
                commit_id=mutation_id,
                fingerprint=fingerprint,
                actor=actor,
                message=f"{plan.mode} {plan.target_commit_id}",
                origin=plan.mode,
                required_access=plan.required_access,
                reverts_commit_id=plan.target_commit_id
                if plan.mode == "revert"
                else None,
                rollback_target_commit_id=plan.target_commit_id
                if plan.mode == "rollback"
                else None,
            )
            query, params = _publish_query(before, after, state=state, receipt=receipt)
            self._execute(pipe, query, params)
            return receipt

    def begin_resource(self, *, pot_id, operation_id, owner):
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            state = _state(self.graph, pot_id, pipe)
            if state is None or state.resource_guard:
                raise JournalError(
                    "journal is absent or resource operation is already incomplete"
                )
            if not operation_id or not owner:
                raise JournalError("resource operation identity is required")
            if _read(
                self.graph,
                pipe,
                f"MATCH (r:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) RETURN r.fingerprint",
                {"pot": pot_id, "mid": f"resource-complete:{operation_id}"},
            ):
                raise JournalError("resource operation ID was reused")
            next_state = replace(
                state,
                resource_generation=state.resource_generation + 1,
                resource_guard=ResourceGuard(
                    operation_id, owner, state.resource_generation + 1
                ),
            )
            query, params = _publish_query({}, {}, state=state, receipt=None)
            params["next_state"] = journal_json(next_state)
            self._execute(pipe, query, params)
            return next_state

    def complete_resource(self, *, pot_id, operation_id, owner):
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            existing = self.get_receipt(
                pot_id=pot_id, commit_id=f"resource-complete:{operation_id}"
            )
            if existing is not None:
                if existing.fingerprint != journal_hash((operation_id, owner)):
                    raise JournalError("resource completion identity was reused")
                return existing
            state = _state(self.graph, pot_id, pipe)
            if (
                state is None
                or state.resource_guard is None
                or (state.resource_guard.operation_id, state.resource_guard.owner)
                != (operation_id, owner)
            ):
                raise JournalError("resource guard ownership mismatch")
            receipt = self._receipt(
                state,
                {},
                {},
                commit_id=f"resource-complete:{operation_id}",
                fingerprint=journal_hash((operation_id, owner)),
                actor=owner,
                message="Resource operation completed",
                origin="resource",
                unsupported_reason="resource content/index restoration is excluded",
            )
            query, params = _publish_query({}, {}, state=state, receipt=receipt)
            next_state = replace(
                state,
                head=receipt.commit_id,
                sequence=receipt.sequence,
                resource_guard=None,
                resource_generation=state.resource_generation + 1,
            )
            params["next_state"] = journal_json(next_state)
            self._execute(pipe, query, params)
            return receipt

    def recover_resource_guard(
        self, *, pot_id, operation_id, expected_owner, new_owner, verify_worker_stopped
    ):
        if (
            not new_owner
            or new_owner == expected_owner
            or verify_worker_stopped(expected_owner) is not True
        ):
            raise JournalError(
                "recovery requires a stopped/fenced original worker and a new owner"
            )
        with transaction_client(self.graph).pipeline() as pipe:
            pipe.watch(self.graph.name)
            state = _state(self.graph, pot_id, pipe)
            if (
                state is None
                or state.resource_guard is None
                or (state.resource_guard.operation_id, state.resource_guard.owner)
                != (operation_id, expected_owner)
            ):
                raise JournalError("resource recovery guard changed")
            next_state = replace(
                state,
                resource_generation=state.resource_generation + 1,
                resource_guard=replace(state.resource_guard, owner=new_owner),
            )
            query, params = _publish_query({}, {}, state=state, receipt=None)
            params["next_state"] = journal_json(next_state)
            self._execute(pipe, query, params)
            return next_state
