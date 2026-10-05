"""Guarded reads plus deterministic exact publication for FalkorDB.

All generated identities/timestamps and normalization are resolved before the
write while WATCH protects the graph key. The exact record changes, next HEAD,
and immutable receipt are published in ONE GRAPH.QUERY. No after-write capture.
"""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

from potpie_context_core.graph_journal import (
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
from potpie_context_core.journal_context import current_journal_context
from potpie_context_core.journal_inverse import plan_inverse, validate_state
from potpie_context_core.reconciliation import MutationResult, MutationSummary
from redis.exceptions import ResponseError, WatchError

from ._mutation_execution import MutationExecutionReuseError
from .falkordb_connection import transaction_client

_STATE = "PotpieRevision"
_RECEIPT = "PotpieMutationReceipt"
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_LIFECYCLE = (
    "invalid_at",
    "expired_at",
    "invalidation_reason",
    "invalidated_by",
    "superseded_by_object",
    "supersession_reason",
    "deleted_by",
)


def _read(graph, pipe, query, params):
    from .falkordb_atomic import _rows

    if pipe is None:
        try:
            return graph.ro_query(query, params=params).result_set
        except ResponseError as exc:
            if "empty key" in str(exc).lower():
                return []
            raise
    return _rows(
        graph,
        pipe.execute_command(
            "GRAPH.RO_QUERY",
            graph.name,
            graph._build_params_header(params) + query,
            "--compact",
        ),
    )


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
        "WHERE $keys IS NULL OR a.entity_key IN $keys OR b.entity_key IN $keys OR r.claim_key IN $keys "
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


def _is_future(value, now):
    value = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise JournalError("native lifecycle timestamps must be timezone aware")
    return value > datetime.fromisoformat(now)


def _effect_plan(before, plan, params, definition):
    """Evaluate the existing compiler's normalized parameters exactly once."""
    after = deepcopy(before)
    entities = {r.logical_key: r for r in after.values() if r.kind == "entity"}
    now = params["now"]
    counts = MutationSummary()
    for i, item in enumerate(plan.entity_upserts):
        old = entities.get(item.entity_key)
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
            value = params[param]
            if value:
                props[prop] = value
            elif not props.get(prop):
                props[prop] = item.entity_key
        props.update(
            entity_key=item.entity_key, group_id=params["pot"], record_id=record_id
        )
        labels = set(old.fields["labels"] if old else ("Entity",))
        labels.difference_update(definition.entity_types)
        from .cypher import coherent_entity_labels

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
            {"active": True, "labels": tuple(sorted(labels)), "properties": props},
        )
        after[record_id] = entities[item.entity_key] = record
        counts.entity_upserts_applied += 1

    def upsert(
        source,
        target,
        predicate,
        source_ref,
        claim_key,
        raw,
        preserve=False,
        generated_valid_at=False,
    ):
        a, b = entities.get(source), entities.get(target)
        if (
            a is None
            or b is None
            or not a.fields.get("active", True)
            or not b.fields.get("active", True)
        ):
            return None
        matches = [
            r
            for r in after.values()
            if r.kind == "claim"
            and r.fields["subject_key"] == source
            and r.fields["object_key"] == target
            and r.fields["predicate"] == predicate
            and r.fields["properties"].get("source_ref") == source_ref
            and r.fields["properties"].get("environment") == raw.get("environment")
            and r.fields.get("active", True)
            and (
                preserve
                or r.fields["properties"].get("invalid_at") is None
                or _is_future(r.fields["properties"]["invalid_at"], now)
            )
            and (not claim_key or r.fields["properties"].get("claim_key") == claim_key)
        ]
        if len(matches) > 1:
            raise JournalError("ambiguous native MERGE incarnation")
        old = matches[0] if matches else None
        props = _props(old) if old else {"uuid": uuid.uuid4().hex, "created_at": now}
        record_id = old.record_id if old else props["uuid"]
        if old is not None and props.get("invalid_at") is None and generated_valid_at:
            raw["valid_at"] = props.get("valid_at", now)
        if props.get("invalid_at") is not None and not preserve:
            props["revived_at"] = now
        if not (preserve and props.get("invalid_at") is not None):
            for key in _LIFECYCLE:
                props.pop(key, None)
        props.update(
            group_id=params["pot"],
            name=predicate,
            subject_key=source,
            object_key=target,
            source_ref=source_ref,
            record_id=record_id,
        )
        if claim_key:
            props["claim_key"] = claim_key
        for key, value in raw.items():
            if value is None:
                props.pop(key, None)
            else:
                props[key] = value
        record = JournalRecord(
            record_id,
            params["pot"],
            props.get("claim_key") or f"{source}:{predicate}:{target}:{source_ref}",
            "claim",
            {
                "active": True,
                "predicate": predicate,
                "subject_key": source,
                "object_key": target,
                "subject_record_id": a.record_id,
                "object_record_id": b.record_id,
                "properties": props,
            },
        )
        after[record_id] = record
        return record

    for i, item in enumerate(plan.edge_upserts):
        raw = deepcopy(params[f"eprop{i}"])
        raw["execution_observed_at"] = raw.pop("observed_at", now)
        if "observed_at" in item.properties:
            raw["observed_at"] = item.properties["observed_at"]
        if params.get("journal_generated_event"):
            raw["execution_source_event_id"] = raw.pop("provenance_source_event", None)
        new = upsert(
            params[f"ef{i}"],
            params[f"et{i}"],
            params[f"en{i}"],
            params[f"es{i}"],
            params[f"eclaim{i}"],
            raw,
            params[f"epreserve{i}"],
            generated_valid_at="valid_at" not in item.properties
            and "valid_from" not in item.properties,
        )
        counts.edge_upserts_applied += new is not None
        if (
            item.edge_type in definition.singleton_predicates
            and params[f"eprop{i}"].get("evidence_strength") == "deterministic"
        ):
            for key, old in list(after.items()):
                if (
                    old.kind == "claim"
                    and old.fields["subject_key"] == item.from_entity_key
                    and old.fields["predicate"] == item.edge_type
                    and old.fields["object_key"] != item.to_entity_key
                    and old.fields["properties"].get("invalid_at") is None
                ):
                    props = _props(old)
                    props.update(
                        invalid_at=params[f"ev{i}"],
                        expired_at=now,
                        superseded_by_object=item.to_entity_key,
                        supersession_reason="singleton_predicate",
                    )
                    after[key] = _replace_props(old, props)
    for i, item in enumerate(plan.edge_deletes):
        for key, old in list(after.items()):
            if (
                old.kind == "claim"
                and old.fields["subject_key"] == item.from_entity_key
                and old.fields["object_key"] == item.to_entity_key
                and old.fields["predicate"] == item.edge_type
            ):
                props = _props(old)
                props.update(invalid_at=now, expired_at=now, deleted_by=params["mid"])
                after[key] = _replace_props(old, props)
                counts.edge_deletes_applied += 1
    for i, item in enumerate(plan.invalidations):
        for key, old in list(after.items()):
            body, props = old.fields, _props(old)
            selected = (
                (
                    item.target_claim_keys is not None
                    and old.kind == "claim"
                    and props.get("claim_key") in item.target_claim_keys
                    and props.get("invalid_at") is None
                )
                or (
                    item.target_entity_key
                    and old.kind == "entity"
                    and old.logical_key == item.target_entity_key
                )
                or (
                    item.target_claim_keys is None
                    and not item.target_entity_key
                    and old.kind == "claim"
                    and (body["predicate"], body["subject_key"], body["object_key"])
                    == item.target_edge
                    and props.get("invalid_at") is None
                )
            )
            if selected:
                props.update(
                    {k: v for k, v in params[f"ip{i}"].items() if v is not None}
                )
                after[key] = _replace_props(old, props)
                counts.invalidations_applied += 1
        if f"isn{i}" in params:
            upsert(
                params[f"isn{i}"],
                params[f"iso{i}"],
                "SUPERSEDES",
                params[f"iss{i}"],
                None,
                params[f"isprop{i}"],
            )
    return after, counts


def _publish_query(before, after, *, state, receipt, result=None):
    params = {
        "pot": state.pot_id,
        "state": journal_json(state),
        "generation": state.generation,
        "old_head": receipt.parent_commit_id if receipt else state.head,
    }
    clauses = [f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.journal=$state"]
    changed = [after[key] for key in after if before.get(key) != after[key]]
    for i, record in enumerate(sorted(changed, key=lambda r: r.kind != "entity")):
        props = _props(record)
        props["retired"] = not record.fields.get("active", True)
        props["record_id"] = record.record_id
        params[f"rid{i}"] = record.record_id
        params[f"props{i}"] = props
        if record.kind == "entity":
            labels = record.fields["labels"]
            old_labels = (
                before[record.record_id].fields["labels"]
                if record.record_id in before
                else ()
            )
            if any(not _LABEL.fullmatch(label) for label in (*labels, *old_labels)):
                raise JournalError("invalid native label")
            remove = sorted(set(old_labels) - set(labels))
            clauses.append(
                f"CALL {{ WITH v MERGE (e:Entity {{group_id:$pot,record_id:$rid{i}}}) "
                f"SET e=$props{i} "
                + (f"REMOVE e:{':'.join(remove)} " if remove else "")
                + (f"SET e:{':'.join(labels)} " if labels else "")
                + f"RETURN count(e) AS count{i} }}"
            )
        else:
            vector = props.pop("fact_embedding", None)
            if vector is not None:
                params[f"vector{i}"] = vector
            params.update(
                {
                    f"src{i}": record.fields["subject_record_id"],
                    f"dst{i}": record.fields["object_record_id"],
                }
            )
            clauses.append(
                f"CALL {{ WITH v MATCH (a:Entity {{group_id:$pot,record_id:$src{i}}}) "
                f"MATCH (b:Entity {{group_id:$pot,record_id:$dst{i}}}) "
                f"MERGE (a)-[r:RELATES_TO {{group_id:$pot,record_id:$rid{i}}}]->(b) SET r=$props{i} "
                + (
                    f"SET r.fact_embedding=vecf32($vector{i}) "
                    if vector is not None
                    else ""
                )
                + f"RETURN count(r) AS count{i} }}"
            )
    next_state = state
    if receipt:
        next_state = replace(state, sequence=receipt.sequence, head=receipt.commit_id)
        params.update(
            mid=receipt.commit_id,
            fingerprint=receipt.fingerprint,
            journal=journal_json(receipt),
            payload=json.dumps(
                {
                    "mutation_summary": {
                        "entity_upserts_applied": result.entity_upserts_applied
                        if result
                        else 0,
                        "edge_upserts_applied": result.edge_upserts_applied
                        if result
                        else 0,
                        "edge_deletes_applied": result.edge_deletes_applied
                        if result
                        else 0,
                        "invalidations_applied": result.invalidations_applied
                        if result
                        else 0,
                        "stamp_counts": {},
                    }
                }
            ),
            entities=result.entity_upserts_applied if result else 0,
            edges=result.edge_upserts_applied if result else 0,
            deletes=result.edge_deletes_applied if result else 0,
            invalidations=result.invalidations_applied if result else 0,
            sequence=receipt.sequence,
        )
        clauses.append(
            f"CREATE (receipt:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid,fingerprint:$fingerprint,"
            "payload:$payload,journal:$journal,journal_generation:$generation,sequence:$sequence,"
            "entities:$entities,edges:$edges,deletes:$deletes,invalidations:$invalidations})"
        )
        clauses.append("SET v.version=v.version+1")
    params["next_state"] = journal_json(next_state)
    clauses.append("SET v.journal=$next_state RETURN v.version")
    return " ".join(clauses), params


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
        from potpie_context_core.journal_capture import capture_receipt

        return capture_receipt(state, before, after, **kwargs)

    def apply(self, plan, *, pipe, state, params, fingerprint, actor):
        context = current_journal_context()
        if context.resource_operation_id and state.resource_guard is None:
            raise JournalError("resource operation no longer owns an active guard")
        if state.resource_guard and (
            context.resource_operation_id,
            context.resource_owner,
        ) != (state.resource_guard.operation_id, state.resource_guard.owner):
            raise JournalError("resource operation owns the pot write guard")
        keys = {item.entity_key for item in plan.entity_upserts}
        keys.update(
            k
            for item in (*plan.edge_upserts, *plan.edge_deletes)
            for k in (item.from_entity_key, item.to_entity_key)
        )
        for item in plan.invalidations:
            keys.update(item.target_claim_keys or ())
            keys.update(
                k for k in (item.target_entity_key, item.superseded_by_key) if k
            )
            if item.target_edge:
                keys.update(item.target_edge[1:])
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
        query, values = _publish_query(
            before, after, state=state, receipt=receipt, result=summary
        )
        self._execute(pipe, query, values)
        return MutationResult(
            True, receipt.commit_id, summary, downgrades=plan.ontology_downgrades
        )

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
                from potpie_context_core.journal_fields import clear_derived

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
