"""Prepare/resume bounded legacy repairs through the existing commit coordinator."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import json
from uuid import NAMESPACE_URL, uuid4, uuid5

from .errors import CapabilityNotImplemented, GraphMutationVersionConflict
from .graph_contract import MutationRisk
from .graph_mutations import ProvenanceContext
from .graph_plans import (
    GraphMutationCommitResult,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
)
from .legacy_records import (
    LegacyInventory,
    RecordRepair,
    MAX_REPAIR_RECORDS,
    REPAIR_ENVELOPE_BYTES,
    repair_effect_bytes,
)
from .legacy_repair_policy import LegacyFinding, plan_legacy_repairs
from .legacy_repair_run import (
    RepairChild,
    RepairRun,
    child_request,
    repair_request_fingerprint,
)
from .mutation_execution import (
    PreparedMutation,
    MutationPersistenceError,
    batch_fingerprint,
    request_fingerprint,
)
from .mutation_state import _subgraph_versions
from .ports.graph.analytics import RepairFinding, RepairReport
from .reconciliation import MutationBatch


def _require_store(store) -> None:
    if not all(
        callable(getattr(store, name, None))
        for name in ("get_repair_run", "reserve_repair_run")
    ):
        raise CapabilityNotImplemented(
            "graph.legacy_repair",
            detail="plan store requires atomic durable manifest/child reservation",
        )


def _children(store, run: RepairRun) -> tuple[GraphMutationPlanRecord, ...]:
    records = tuple(
        store.get(pot_id=run.pot_id, plan_id=child.plan_id) for child in run.children
    )
    if any(record is None for record in records):
        raise MutationPersistenceError("repair child is missing")
    run.validate_children(records)
    return records


def _pack(
    effects: tuple[RecordRepair, ...], inventory: LegacyInventory, chunk_size: int
):
    max_records = inventory.max_repair_records
    max_bytes = inventory.max_repair_bytes
    chunks, current, size = [], [], REPAIR_ENVELOPE_BYTES
    if (
        type(max_records) is not int
        or max_records < 1
        or type(max_bytes) is not int
        or max_bytes < REPAIR_ENVELOPE_BYTES
    ):
        return (), (
            LegacyFinding(
                "unsupported_limits",
                "Backend repair limits cannot fit a bounded child.",
            ),
        )
    max_records = min(chunk_size, MAX_REPAIR_RECORDS, max_records)
    for effect in effects:
        # Allow for the raw before+after images, journal JSON encoding and both
        # execution/journal envelopes. Backends enforce their exact native limits.
        cost = repair_effect_bytes(effect)
        if cost + REPAIR_ENVELOPE_BYTES > max_bytes:
            return (), (
                LegacyFinding(
                    "oversized_record",
                    "A repair record exceeds the backend transaction limit.",
                    effect.before.fingerprint,
                ),
            )
        if current and (len(current) >= max_records or size + cost > max_bytes):
            chunks.append(tuple(current))
            current, size = [], REPAIR_ENVELOPE_BYTES
        current.append(effect)
        size += cost
    if current:
        chunks.append(tuple(current))
    return tuple(chunks), ()


def prepare_legacy_repair(
    backend, store, *, pot_id: str, repair_id: str | None = None, chunk_size: int = 100
) -> RepairReport:
    _require_store(store)
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_REPAIR_RECORDS:
        raise ValueError("invalid repair chunk size")
    repair_id = repair_id or uuid4().hex
    fingerprint = repair_request_fingerprint(chunk_size=chunk_size)
    existing = store.get_repair_run(pot_id=pot_id, repair_id=repair_id)
    if existing is not None:
        if existing.request_fingerprint != fingerprint:
            raise MutationPersistenceError(
                "repair identity is bound to a different request"
            )
        return _report(existing, _children(store, existing))
    inventory_read = getattr(backend.mutation, "legacy_inventory", None)
    if not callable(inventory_read):
        raise CapabilityNotImplemented(
            "graph.legacy_repair", detail="backend has no lossless legacy inventory"
        )
    before = _subgraph_versions(backend, pot_id)["_global"]
    inventory = inventory_read(pot_id)
    after = _subgraph_versions(backend, pot_id)["_global"]
    if before != after:
        raise GraphMutationVersionConflict(expected=before, current=after)
    if not isinstance(inventory, LegacyInventory) or any(
        record.pot_id != pot_id for record in inventory.records
    ):
        raise MutationPersistenceError("invalid repair inventory scope")
    effects, findings = plan_legacy_repairs(inventory)
    chunks, limit_findings = _pack(effects, inventory, chunk_size)
    findings += limit_findings
    if any(finding.blocking for finding in findings):
        chunks = ()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=24)
    children, bindings = [], []
    for index, chunk in enumerate(chunks):
        identity = str(
            uuid5(
                NAMESPACE_URL,
                json.dumps(["potpie:legacy-repair", pot_id, repair_id, index]),
            )
        )
        plan_id, mutation_id = f"repair-plan:{identity}", f"repair-mutation:{identity}"
        batch = MutationBatch(record_repairs=chunk)
        payload = child_request(repair_id, index)
        prepared = PreparedMutation(
            pot_id,
            mutation_id,
            request_fingerprint(payload),
            batch_fingerprint(batch),
            before + index,
            now,
            tuple(
                effect.after().value("record_id")
                for effect in chunk
                if effect.before.kind == "claim"
            ),
        )
        child = GraphMutationPlanRecord(
            plan_id,
            pot_id,
            GraphMutationPlanStatus.review_required.value,
            MutationRisk.high.value,
            now,
            expires,
            original_payload=payload,
            lowered_batch=batch,
            provenance=ProvenanceContext(mutation_id=mutation_id, prepared=prepared),
            expected_subgraph_versions={"_global": before + index},
            mutation_id=mutation_id,
            prepared=prepared,
        )
        children.append(child)
        bindings.append(
            RepairChild(
                plan_id, mutation_id, prepared.batch_fingerprint, before + index
            )
        )
    run = RepairRun(
        repair_id,
        pot_id,
        fingerprint,
        before,
        now,
        expires,
        tuple(bindings),
        findings,
        len(inventory.records),
        len(effects),
    )
    run, _ = store.reserve_repair_run(run, tuple(children))
    return _report(run, _children(store, run))


def apply_legacy_repair(
    store,
    *,
    pot_id: str,
    repair_id: str,
    commit_child: Callable[[RepairRun, RepairChild], GraphMutationCommitResult],
) -> RepairReport:
    _require_store(store)
    run = store.get_repair_run(pot_id=pot_id, repair_id=repair_id)
    if run is None:
        return RepairReport(
            pot_id,
            ("legacy",),
            ok=False,
            status="not_found",
            repair_id=repair_id,
            detail="Prepare a legacy repair before applying it.",
        )
    records = _children(store, run)
    if run.blocked:
        return _report(run, records)
    executions = []
    completed = 0
    for child in run.children:
        # Existing coordinator checks receipts before expiry/revision/attempt
        # takeover. There is no repair-specific apply or recovery implementation.
        result = commit_child(run, child)
        if not result.ok:
            return _report(
                run,
                records,
                completed=completed,
                executions=tuple(executions),
                status=result.status,
                detail=result.detail,
                ok=False,
            )
        if result.execution is None:
            raise MutationPersistenceError(
                "repair completion has no authoritative execution metadata"
            )
        executions.append(
            {"mutation_id": result.mutation_id, **result.execution.to_dict()}
        )
        completed += 1
    return _report(
        run,
        records,
        completed=completed,
        executions=tuple(executions),
        status="completed",
    )


def require_repair_execution(
    store, record: GraphMutationPlanRecord, authorization: RepairRun | None
) -> None:
    """Generic graph.commit cannot execute an administrative repair child."""
    if not record.lowered_batch or not record.lowered_batch.record_repairs:
        return
    if authorization is None or authorization.pot_id != record.pot_id:
        raise PermissionError(
            "legacy repair children require the administrative repair apply path"
        )
    stored = store.get_repair_run(
        pot_id=record.pot_id, repair_id=authorization.repair_id
    )
    if stored != authorization or stored.blocked:
        raise MutationPersistenceError(
            "repair authorization differs from reviewed manifest"
        )
    records = _children(store, stored)
    if not any(child.plan_id == record.plan_id for child in records):
        raise MutationPersistenceError("plan is not a member of the reviewed repair")
    # The ordinary coordinator checks the bound child's identical expiry next.


def _report(
    run: RepairRun,
    records,
    *,
    completed: int = 0,
    executions=(),
    status: str | None = None,
    detail: str | None = None,
    ok: bool = True,
) -> RepairReport:
    effects = [
        effect for record in records for effect in record.lowered_batch.record_repairs
    ]
    completed_effects = [
        effect
        for record in records[:completed]
        for effect in record.lowered_batch.record_repairs
    ]
    return RepairReport(
        run.pot_id,
        ("legacy",),
        repaired={
            kind: sum(effect.before.kind == kind for effect in completed_effects)
            for kind in ("entity", "claim")
        },
        detail=detail
        or (
            "Resolve blocking findings and prepare a fresh repair."
            if run.blocked
            else (
                "Legacy repair completed. Existing erased history remains unavailable."
                if status == "completed"
                else "Review this frozen repair before applying it. Existing erased history remains unavailable."
            )
        ),
        findings=tuple(
            RepairFinding(
                finding.code,
                1,
                (finding.record_fingerprint,) if finding.record_fingerprint else (),
                finding.detail,
                (
                    "Resolve explicitly, then prepare a fresh repair."
                    if finding.blocking
                    else None
                ),
            )
            for finding in run.findings
        ),
        ok=ok and not run.blocked,
        status=status or ("blocked" if run.blocked else "ready"),
        repair_id=run.repair_id,
        scanned_records=run.scanned_records,
        repairable_records=run.repairable_records,
        completed_batches=completed,
        total_batches=len(run.children),
        expected_revision=run.base_revision,
        expires_at=run.expires_at.isoformat(),
        changes=tuple(
            {
                "kind": effect.before.kind,
                "key": effect.before.key,
                "before_fingerprint": effect.before.fingerprint,
                "field_updates": dict(effect.field_updates),
                "property_updates": dict(effect.property_updates),
            }
            for effect in effects
        ),
        executions=executions,
    )
