"""Reference journal transactions over the current in-memory store.

Embedded runs these transactions on an isolated reload, then publishes the
store, guards, revision, and receipts with one file replacement.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timezone
from itertools import islice

from potpie_context_engine.core.graph_journal import (
    CommitReceipt,
    JournalCapability,
    JournalError,
    JournalLimits,
    JournalRecord,
    JournalState,
    ResourceGuard,
    journal_hash,
)
from potpie_context_engine.core.journal_context import current_journal_context
from potpie_context_engine.core.journal_inverse import (
    plan_inverse,
    protocols_enabled_for,
    validate_state,
)
from potpie_context_engine.core.ports.claim_query import ClaimRow

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    mutation_batch_fingerprint,
)


def migrate_ids(store, pot_id: str) -> None:
    """Assign identity under the caller's write guard; reject duplicate stored IDs."""
    keys = {key for pid, key in store.entity_label_index if pid == pot_id}
    keys.update(key for pid, key in store.entity_property_index if pid == pot_id)
    keys.update(
        key
        for r in store.rows
        if r.pot_id == pot_id
        for key in (r.subject_key, r.object_key)
    )
    used = set()
    for key in sorted(keys):
        props = store.entity_property_index.setdefault((pot_id, key), {})
        record_id = props.get("record_id") or props.get("uuid") or uuid.uuid4().hex
        if not isinstance(record_id, str) or record_id in used:
            raise JournalError("ambiguous entity record identity")
        used.add(record_id)
        props["record_id"] = record_id
        store.entity_label_index.setdefault((pot_id, key), ("Entity",))
    for i, row in enumerate(store.rows):
        if row.pot_id != pot_id:
            continue
        record_id = row.record_id or uuid.uuid4().hex
        if not isinstance(record_id, str) or record_id in used:
            raise JournalError("ambiguous claim record identity")
        used.add(record_id)
        store.rows[i] = replace(row, record_id=record_id)


def records_from_store(store, pot_id: str) -> dict[str, JournalRecord]:
    records = {}
    for (pid, key), properties in store.entity_property_index.items():
        if pid != pot_id:
            continue
        record_id = properties["record_id"]
        records[record_id] = JournalRecord(
            record_id,
            pot_id,
            key,
            "entity",
            {
                "active": not properties.get("retired", False),
                "labels": tuple(sorted(store.entity_label_index.get((pid, key), ()))),
                "properties": {
                    k: deepcopy(v) for k, v in properties.items() if k != "retired"
                },
            },
        )
    for row in store.rows:
        if row.pot_id != pot_id:
            continue
        body = {
            f.name: deepcopy(getattr(row, f.name))
            for f in fields(row)
            if f.name not in {"pot_id", "record_id", "retired"}
        }
        body["active"] = not row.retired
        for name, key in (
            ("subject_record_id", row.subject_key),
            ("object_record_id", row.object_key),
        ):
            body[name] = store.entity_property_index[(pot_id, key)]["record_id"]
        records[row.record_id] = JournalRecord(
            row.record_id,
            pot_id,
            row.claim_key
            or f"{row.subject_key}:{row.predicate}:{row.object_key}:{row.source_ref}",
            "claim",
            body,
        )
    return records


def publish_records(store, records, *, mutation_id: str) -> None:
    """Replace exact typed state while advancing execution stamps."""
    row_fields = {f.name for f in fields(ClaimRow)}
    for record in records:
        from potpie_context_engine.core.journal_fields import clear_derived

        record = clear_derived(record)
        body = dict(record.fields)
        if record.kind == "entity":
            properties = deepcopy(dict(body["properties"]))
            properties["record_id"] = record.record_id
            properties["retired"] = not body.get("active", True)
            properties["prov_mutation_id"] = mutation_id
            properties["prov_graph_updated_at"] = datetime.now(timezone.utc).isoformat()
            store.entity_property_index[(record.pot_id, record.logical_key)] = (
                properties
            )
            store.entity_label_index[(record.pot_id, record.logical_key)] = tuple(
                body["labels"]
            )
        else:
            values = {k: deepcopy(v) for k, v in body.items() if k in row_fields}
            values.update(
                pot_id=record.pot_id,
                record_id=record.record_id,
                retired=not body.get("active", True),
                mutation_id=mutation_id,
            )
            values["properties"]["mutation_id"] = mutation_id
            row = ClaimRow(**values)
            matches = [
                i
                for i, r in enumerate(store.rows)
                if r.pot_id == record.pot_id and r.record_id == record.record_id
            ]
            if len(matches) != 1:
                raise JournalError("restore requires exactly one stored incarnation")
            store.rows[matches[0]] = row


def adopt_store(target, source) -> None:
    target.rows[:] = source.rows
    for name in ("entity_property_index", "entity_label_index"):
        current = getattr(target, name)
        current.clear()
        current.update(getattr(source, name))


class MemoryJournal:
    def __init__(self, mutation):
        self.owner = mutation

    def journal_capability(self, pot_id: str) -> JournalCapability:
        state = self.journal_state(pot_id)
        return JournalCapability(
            True,
            bool(state and state.rollback_enabled),
            self.owner.profile == "embedded",
            "volatile" if self.owner.profile == "in_memory" else None,
        )

    def journal_state(self, pot_id: str) -> JournalState | None:
        with self.owner.state_lock:
            return deepcopy(
                self.owner.journal_data.setdefault("states", {}).get(pot_id)
            )

    def activate(
        self,
        *,
        pot_id: str,
        rollback_enabled: bool = False,
        limits: JournalLimits | None = None,
    ) -> JournalState:
        with self.owner.state_lock:
            state = self.journal_state(pot_id)
            if state:
                return state
            staged = clone_store(self.owner.store)
            migrate_ids(staged, pot_id)
            state = JournalState(
                pot_id,
                uuid.uuid4().hex,
                rollback_enabled=rollback_enabled,
                limits=limits or JournalLimits(),
            )
            adopt_store(self.owner.store, staged)
            self.owner.journal_data["states"][pot_id] = state
            self.owner.journal_data.setdefault("receipts", {})[pot_id] = []
            self.owner._notify()
            return deepcopy(state)

    def set_rollback_enabled(self, *, pot_id: str, enabled: bool) -> JournalState:
        with self.owner.state_lock:
            state = self.journal_state(pot_id)
            if state is None:
                raise JournalError("journal is not active")
            updated = replace(state, rollback_enabled=enabled)
            self.owner.journal_data["states"][pot_id] = updated
            self.owner._notify()
            return deepcopy(updated)

    def get_receipt(self, *, pot_id: str, commit_id: str) -> CommitReceipt | None:
        with self.owner.state_lock:
            return deepcopy(
                next(
                    (
                        r
                        for r in self.owner.journal_data.get("receipts", {}).get(
                            pot_id, ()
                        )
                        if r.commit_id == commit_id
                    ),
                    None,
                )
            )

    def read_receipts(
        self, *, pot_id: str, generation: str, after_sequence: int, limit: int = 100
    ):
        with self.owner.state_lock:
            if not 1 <= limit <= 200:
                raise JournalError("receipt page size must be between 1 and 200")
            return tuple(
                deepcopy(r)
                for r in islice(
                    (
                        r
                        for r in self.owner.journal_data.get("receipts", {}).get(
                            pot_id, ()
                        )
                        if r.journal_generation == generation
                        and r.sequence > after_sequence
                    ),
                    limit,
                )
            )

    def _guard_write(self, state):
        context = current_journal_context()
        guard = state.resource_guard
        if context.resource_operation_id and guard is None:
            raise JournalError("resource operation no longer owns an active guard")
        if guard and (context.resource_operation_id, context.resource_owner) != (
            guard.operation_id,
            guard.owner,
        ):
            raise JournalError("resource operation owns the pot write guard")

    def _receipt(self, state, before, after, **kwargs):
        from potpie_context_engine.core.journal_capture import capture_receipt

        return capture_receipt(state, before, after, **kwargs)

    def _append(self, state, receipt):
        if (
            self.get_receipt(pot_id=state.pot_id, commit_id=receipt.commit_id)
            is not None
        ):
            raise JournalError("commit ID was reused")
        self.owner.journal_data["receipts"][state.pot_id].append(deepcopy(receipt))
        self.owner.journal_data["states"][state.pot_id] = replace(
            state, sequence=receipt.sequence, head=receipt.commit_id
        )
        self.owner.revisions[state.pot_id] = (
            self.owner.revisions.get(state.pot_id, 0) + 1
        )

    def apply(self, plan, *, original_plan, pot_id, mutation_id, provenance):
        from .backends.in_memory_backend import _Mutation

        state = self.journal_state(pot_id)
        assert state is not None
        receipt = self.get_receipt(pot_id=pot_id, commit_id=mutation_id)
        if receipt is not None and receipt.fingerprint != mutation_batch_fingerprint(
            original_plan
        ):
            raise JournalError("commit ID was reused with a different mutation")
        prior = self.owner.execution_registry.lookup(
            original_plan, expected_pot_id=pot_id, mutation_id=mutation_id
        )
        if prior.result is not None:
            return prior.result
        self._guard_write(state)
        staged = clone_store(self.owner.store)
        before = records_from_store(staged, pot_id)
        writer = _Mutation(
            staged,
            definition=self.owner.definition,
            embedder=self.owner.embedder,
            journal_data=deepcopy(self.owner.journal_data),
        )
        result = writer._apply_uncached(
            plan, expected_pot_id=pot_id, mutation_id=mutation_id
        )
        for ent in plan.entity_upserts:
            props = staged.entity_property_index[(pot_id, ent.entity_key)]
            old = self.owner.store.entity_property_index.get(
                (pot_id, ent.entity_key), {}
            )
            if old.get("record_id"):
                props["record_id"] = old["record_id"]
            if old.get("uuid"):
                props["uuid"] = old["uuid"]
            props["retired"] = False
        migrate_ids(staged, pot_id)
        after = records_from_store(staged, pot_id)
        receipt = self._receipt(
            state,
            before,
            after,
            commit_id=mutation_id,
            fingerprint=mutation_batch_fingerprint(original_plan),
            actor=provenance.actor_user_id
            if provenance and provenance.actor_user_id
            else "system",
            message=plan.summary,
        )

        def publish():
            adopt_store(self.owner.store, staged)
            self._append(state, receipt)
            return result

        return self.owner.execution_registry.execute(
            original_plan,
            expected_pot_id=pot_id,
            mutation_id=mutation_id,
            operation=publish,
            on_completed=self.owner._notify,
        )

    def plan_restore(self, *, pot_id, target_commit_id, expected_head, mode="revert"):
        with self.owner.state_lock:
            state = self.journal_state(pot_id)
            if state is None:
                raise JournalError("journal is not active")
            return plan_inverse(
                state=state,
                current=records_from_store(self.owner.store, pot_id),
                receipts=self.owner.journal_data["receipts"][pot_id],
                target_commit_id=target_commit_id,
                expected_head=expected_head,
                mode=mode,
                validator=lambda records: validate_state(
                    records,
                    pot_id=pot_id,
                    definition=self.owner.definition,
                    protocols_enabled=protocols_enabled_for(self.owner.definition),
                    resource_exists=(
                        lambda ref: self.owner.resource_exists(pot_id, ref)
                    )
                    if self.owner.resource_exists
                    else None,
                ),
            )

    def apply_restore(self, plan, *, mutation_id: str, actor: str):
        with self.owner.state_lock:
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
            prior = self.get_receipt(pot_id=plan.pot_id, commit_id=mutation_id)
            if prior:
                if prior.fingerprint != fingerprint:
                    raise JournalError("commit ID was reused with a different restore")
                return prior
            if any(
                r.pot_id == plan.pot_id and r.mutation_id == mutation_id
                for r in self.owner.execution_registry.completed()
            ):
                raise JournalError("legacy mutation ID was reused for a restore")
            fresh = self.plan_restore(
                pot_id=plan.pot_id,
                target_commit_id=plan.target_commit_id,
                expected_head=plan.expected_head,
                mode=plan.mode,
            )
            if fresh != plan:
                raise JournalError("restore plan or resource generation changed")
            state = self.journal_state(plan.pot_id)
            staged = clone_store(self.owner.store)
            before = records_from_store(staged, plan.pot_id)
            publish_records(
                staged, (p.after for p in plan.records), mutation_id=mutation_id
            )
            after = records_from_store(staged, plan.pot_id)
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
            adopt_store(self.owner.store, staged)
            self._append(state, receipt)
            self.owner._notify()
            return deepcopy(receipt)

    def begin_resource(self, *, pot_id: str, operation_id: str, owner: str):
        with self.owner.state_lock:
            state = self.journal_state(pot_id)
            if state is None:
                raise JournalError("journal is not active")
            if state.resource_guard:
                raise JournalError("resource operation is already active or incomplete")
            if not operation_id or not owner:
                raise JournalError("resource operation identity is required")
            if self.get_receipt(
                pot_id=pot_id, commit_id=f"resource-complete:{operation_id}"
            ) or any(
                r.pot_id == pot_id
                and r.mutation_id == f"resource-complete:{operation_id}"
                for r in self.owner.execution_registry.completed()
            ):
                raise JournalError("resource operation ID was reused")
            state = replace(
                state,
                resource_generation=state.resource_generation + 1,
                resource_guard=ResourceGuard(
                    operation_id, owner, state.resource_generation + 1
                ),
            )
            self.owner.journal_data["states"][pot_id] = state
            self.owner._notify()
            return deepcopy(state)

    def complete_resource(self, *, pot_id: str, operation_id: str, owner: str):
        with self.owner.state_lock:
            existing = self.get_receipt(
                pot_id=pot_id, commit_id=f"resource-complete:{operation_id}"
            )
            if existing is not None:
                if existing.fingerprint != journal_hash((operation_id, owner)):
                    raise JournalError("resource completion identity was reused")
                return existing
            state = self.journal_state(pot_id)
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
            self._append(state, receipt)
            self.owner.journal_data["states"][pot_id] = replace(
                self.owner.journal_data["states"][pot_id],
                resource_guard=None,
                resource_generation=state.resource_generation + 1,
            )
            self.owner._notify()
            return deepcopy(receipt)

    def recover_resource_guard(
        self, *, pot_id, operation_id, expected_owner, new_owner, verify_worker_stopped
    ):
        """Require proof from the execution host, never lease expiry."""
        if (
            not new_owner
            or new_owner == expected_owner
            or verify_worker_stopped(expected_owner) is not True
        ):
            raise JournalError(
                "recovery requires a stopped/fenced original worker and a new owner"
            )
        with self.owner.state_lock:
            state = self.journal_state(pot_id)
            if (
                state is None
                or state.resource_guard is None
                or (state.resource_guard.operation_id, state.resource_guard.owner)
                != (operation_id, expected_owner)
            ):
                raise JournalError("resource recovery guard changed")
            state = replace(
                state,
                resource_generation=state.resource_generation + 1,
                resource_guard=replace(state.resource_guard, owner=new_owner),
            )
            self.owner.journal_data["states"][pot_id] = state
            self.owner._notify()
            return deepcopy(state)


def repair_journaled(analytics, pot_id, targets):
    from potpie_context_engine.core.graph_mutations import EntityUpsert
    from potpie_context_engine.core.journal_context import (
        JournalWriteContext,
        journal_write_context,
    )
    from potpie_context_engine.core.ports.graph.analytics import RepairReport
    from potpie_context_engine.core.reconciliation import MutationBatch

    from .entity_label_repair import repaired_entity_labels, wants_entity_label_repair
    from .entity_summary_repair import (
        repaired_entity_properties,
        wants_entity_summary_repair,
    )

    with analytics.mutation.state_lock:
        operations, repaired = [], {"entity_summaries": 0, "entity_labels": 0}
        for (pid, key), properties in analytics.store.entity_property_index.items():
            if pid != pot_id or properties.get("retired", False):
                continue
            labels = analytics.store.entity_label_index.get((pid, key), ())
            props = (
                repaired_entity_properties(key, properties)
                if wants_entity_summary_repair(targets)
                else None
            )
            fixed_labels = (
                repaired_entity_labels(
                    key, labels, entity_types=analytics.definition.entity_types
                )
                if wants_entity_label_repair(targets)
                else None
            )
            if props is not None or fixed_labels is not None:
                operations.append(
                    EntityUpsert(key, tuple(fixed_labels or labels), props or {})
                )
                repaired["entity_summaries"] += props is not None
                repaired["entity_labels"] += fixed_labels is not None
        if operations:
            with journal_write_context(
                JournalWriteContext(origin="repair", required_access="admin")
            ):
                analytics.mutation.apply(
                    MutationBatch(
                        summary="Repair entity summaries/labels",
                        entity_upserts=operations,
                    ),
                    expected_pot_id=pot_id,
                )
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired=repaired,
            detail="semantic repairs captured in graph journal",
        )


def clone_store(store):
    from .in_memory_reader import InMemoryClaimQueryStore

    return InMemoryClaimQueryStore(
        rows=deepcopy(store.rows),
        entity_label_index=deepcopy(store.entity_label_index),
        entity_property_index=deepcopy(store.entity_property_index),
        embedder=store.embedder,
    )
