"""Mutation identity, revision and workflow-state helpers."""

from __future__ import annotations

from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    MutationCapabilities,
    batch_fingerprint,
    execution_matches_batch,
)

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Callable

from potpie_context_engine.core.errors import (
    CapabilityNotImplemented,
    GraphMutationVersionConflict,
)
from potpie_context_engine.core.graph_contract import (
    MutationRisk,
)
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationResult
from potpie_context_engine.core.graph_plans import (
    GraphMutationCommitResult,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
    GraphMutationProposal,
)
from potpie_context_engine.core.ports.claim_query import (
    ClaimRow,
)
from potpie_context_engine.core.ports.graph.backend import GraphBackend
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.ports.graph.plan_store import GraphPlanStorePort
from potpie_context_engine.core.reconciliation_config import (
    ReconciliationConfig,
    reconciliation_config_scope,
)
from potpie_context_engine.core.semantic_mutations import (
    LoweredOperation,
    SemanticMutationValidationIssue,
)

_LOG = logging.getLogger(__name__)

_CONFLICT_NEXT_ACTION = (
    "Use a fresh idempotency_key and re-run `potpie graph propose`, then commit the new "
    "plan_id; nothing from this plan was applied."
)


_CONFLICT_LANDED_SHOWN = 5


def _subgraph_versions(backend: GraphBackend, pot_id: str) -> dict[str, int]:
    current_version = getattr(backend.mutation, "current_version", None)
    capabilities = getattr(backend.mutation, "capabilities", MutationCapabilities())
    if (
        not capabilities.revision_check
        or not capabilities.atomic_writes
        or not callable(current_version)
    ):
        raise CapabilityNotImplemented(
            "atomic_mutation", detail="backend must expose authoritative revisions"
        )
    return {"_global": int(current_version(pot_id))}


def _apply_if_current(
    backend: GraphBackend,
    *,
    expected_versions: Mapping[str, int],
    atomic_apply: Callable[[int], Any] | None,
    reconciliation_config: ReconciliationConfig,
) -> tuple[Any | None, dict[str, int]]:
    """Use the backend's atomic compare-and-apply boundary exclusively."""
    capabilities = getattr(backend.mutation, "capabilities", MutationCapabilities())
    if (
        not capabilities.atomic_writes
        or not capabilities.revision_check
        or atomic_apply is None
    ):
        raise CapabilityNotImplemented(
            "atomic_mutation", detail="backend must implement atomic compare-and-apply"
        )
    expected = int(expected_versions["_global"])
    try:
        with reconciliation_config_scope(reconciliation_config):
            result = atomic_apply(expected)
    except GraphMutationVersionConflict as exc:
        return None, {"_global": exc.current}
    return result, {"_global": expected}


def _plan_for_idempotency_key(
    store: GraphPlanStorePort, *, pot_id: str, idempotency_key: str
) -> GraphMutationPlanRecord | None:
    for record in store.list(pot_id=pot_id):
        if (
            record.reserves_idempotency
            and str(record.original_payload.get("idempotency_key") or "").strip()
            == idempotency_key
        ):
            return record
    return None


def _expected_versions(
    payload: Mapping[str, Any],
    current_versions: Mapping[str, int],
) -> dict[str, int]:
    provided = _mapping_of_ints(payload.get("expected_subgraph_versions"))
    if "_global" not in provided and "_global" in current_versions:
        provided["_global"] = int(current_versions["_global"])
    return provided


def _version_conflict(
    expected: Mapping[str, int],
    current: Mapping[str, int],
) -> dict[str, Any] | None:
    for key, expected_value in expected.items():
        if key not in current:
            continue
        actual = int(current[key])
        if int(expected_value) != actual:
            return {
                "subgraph": key,
                "expected_version": int(expected_value),
                "actual_version": actual,
            }
    return None


def _conflict_message(
    conflict: Mapping[str, Any],
    *,
    landed: Sequence[str] = (),
    landed_count: int | None = None,
) -> str:
    """Say what moved the pinned version and, when known, what moved it.

    "changed after the plan was proposed (expected 17, actual 28)" told the
    caller nothing about *why*: a plan pins the whole pot's version, so a
    ``record``, a resource import, or a hook's nudge landing in between is
    enough. Naming the commits and the rule turns a retry-and-hope into a
    one-line repair.
    """
    message = (
        f"{conflict.get('subgraph')} moved from {conflict.get('expected_version')} "
        f"to {conflict.get('actual_version')} after this plan was proposed"
    )
    if landed:
        count = landed_count if landed_count is not None else len(landed)
        shown = ", ".join(landed)
        more = f" (+{count - len(landed)} more)" if count > len(landed) else ""
        message += f"; {count} commit(s) landed in between: {shown}{more}"
    return (
        f"{message}. Plans pin the whole pot's version, so any commit in between "
        "— another plan, a `potpie record`, a resource import — conflicts; "
        "nothing from this plan was applied."
    )


def _commits_landed_since(
    plan_store: GraphPlanStorePort,
    *,
    pot_id: str,
    since: datetime,
    exclude_plan_id: str,
    limit: int = _CONFLICT_LANDED_SHOWN,
) -> tuple[tuple[str, ...], int]:
    """Plan ids committed to ``pot_id`` at or after ``since``, oldest first.

    Best effort: the conflict is already decided and this only makes its
    message specific, so a store that cannot list answers "unknown" rather
    than failing the commit.
    """
    try:
        records = plan_store.list(pot_id=pot_id)
        landed = sorted(
            (
                record
                for record in records
                if record.plan_id != exclude_plan_id
                and record.status == GraphMutationPlanStatus.committed.value
                and record.committed_at is not None
                and record.committed_at >= since
            ),
            key=lambda record: record.committed_at or since,
        )
    except Exception:  # noqa: BLE001 - diagnostics must not fail the commit
        return (), 0
    return tuple(record.plan_id for record in landed[:limit]), len(landed)


def _commit_next_action(plan_id: str, *, approved: bool) -> str:
    """The commit command that will actually go through for this plan.

    Names ``--approved-by`` (the flag ``graph commit`` has) rather than
    ``--allow-review-required`` (a flag only the legacy ``graph mutate``
    takes), and includes ``--verify`` so the write is read back.
    """
    if approved:
        return f"Commit with `potpie graph commit {plan_id} --verify`."
    return (
        f"Review the plan, then run `potpie graph commit {plan_id} "
        "--approved-by <user-ref> --verify`."
    )


def _status_for_proposal(semantic_plan, *, conflict: bool) -> str:
    if conflict:
        return GraphMutationPlanStatus.conflict.value
    if semantic_plan.errors or semantic_plan.decision == "rejected":
        return GraphMutationPlanStatus.invalid.value
    if semantic_plan.decision == "review_required":
        return GraphMutationPlanStatus.review_required.value
    return GraphMutationPlanStatus.validated.value


def _lowered_operation_summary(op: LoweredOperation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "op_index": op.op_index,
        "op": op.op,
        "risk": op.risk,
        "status": op.status,
        "claim_keys": list(op.claim_keys),
    }
    if op.subgraph:
        out["subgraph"] = op.subgraph
    if op.truth:
        out["truth"] = op.truth
    return out


def _rejected_operation_summaries(
    issues: tuple[SemanticMutationValidationIssue, ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "op_index": issue.op_index,
            "code": issue.code,
            "message": issue.message,
        }
        for issue in issues
        if issue.is_error
    )


def _proposal_from_record(
    record: GraphMutationPlanRecord,
    *,
    ok: bool,
    recommended_next_action: str | None = None,
) -> GraphMutationProposal:
    claim_keys = _claim_keys_from_record(record)
    return GraphMutationProposal(
        ok=ok,
        plan_id=record.plan_id,
        status=record.status,
        risk=record.risk,
        pot_id=record.pot_id,
        approval=record.approval,
        auto_applicable=(
            record.status == GraphMutationPlanStatus.validated.value
            and record.risk == MutationRisk.low.value
        ),
        expires_at=record.expires_at,
        expected_subgraph_versions=record.expected_subgraph_versions,
        current_subgraph_versions=record.current_subgraph_versions,
        diff=record.diff,
        warnings=record.warnings,
        issues=record.validation_issues,
        rejected_operations=record.rejected_ops,
        review_required_operations=record.review_required_ops,
        claim_keys=claim_keys,
        recommended_next_action=recommended_next_action,
        detail=record.detail,
    )


def _lookup_mutation_execution(
    backend: GraphBackend,
    record: GraphMutationPlanRecord,
) -> MutationExecutionLookup:
    """Query optional backend apply-once state without assuming support."""

    mutation_id = record.mutation_id or ""
    unsupported = MutationExecutionLookup(
        state=MutationExecutionState.unsupported.value,
        mutation_id=mutation_id,
        batch_fingerprint="",
    )
    if not mutation_id or record.lowered_batch is None:
        return unsupported
    lookup_method = getattr(backend.mutation, "lookup_execution", None)
    capabilities = getattr(backend.mutation, "capabilities", MutationCapabilities())
    if not capabilities.execution_lookup or not callable(lookup_method):
        return unsupported
    try:
        lookup = lookup_method(
            record.lowered_batch,
            expected_pot_id=record.pot_id,
            mutation_id=mutation_id,
        )
    except (AttributeError, CapabilityNotImplemented):
        return unsupported
    except Exception:
        _LOG.exception(
            "graph backend execution lookup failed "
            "(pot_id=%s, plan_id=%s, mutation_id=%s)",
            record.pot_id,
            record.plan_id,
            mutation_id,
        )
        return MutationExecutionLookup(
            state=MutationExecutionState.ambiguous.value,
            mutation_id=mutation_id,
            batch_fingerprint="",
        )
    if not isinstance(lookup, MutationExecutionLookup):
        return MutationExecutionLookup(
            state=MutationExecutionState.ambiguous.value,
            mutation_id=mutation_id,
            batch_fingerprint="",
        )
    invalid = lookup.mutation_id != mutation_id
    if lookup.state == MutationExecutionState.completed.value:
        invalid = invalid or (
            lookup.batch_fingerprint != batch_fingerprint(record.lowered_batch)
            or not _execution_matches_record(record, lookup.result)
        )
    if invalid:
        return replace(
            lookup,
            state=MutationExecutionState.ambiguous.value,
            result=None,
            detail=None,
        )
    return lookup


def _execution_matches_record(
    record: GraphMutationPlanRecord, result: MutationResult | None
) -> bool:
    return (
        record.lowered_batch is not None
        and execution_matches_batch(
            result,
            record.lowered_batch,
            pot_id=record.pot_id,
            mutation_id=record.mutation_id or "",
            prepared=record.prepared,
            allow_legacy=record.execution is None,
        )
        and (record.execution is None or record.execution == result.execution)
    )


def _committed_plan_result(
    record: GraphMutationPlanRecord,
) -> GraphMutationCommitResult:
    """Reconstruct an idempotent commit receipt from the persisted plan."""

    return GraphMutationCommitResult(
        ok=True,
        plan_id=record.plan_id,
        status=record.status,
        risk=record.risk,
        pot_id=record.pot_id,
        mutation_id=record.mutation_id,
        applied_at=record.committed_at,
        execution=record.execution,
        expected_subgraph_versions=record.expected_subgraph_versions,
        current_subgraph_versions=record.current_subgraph_versions,
        new_subgraph_versions=record.final_subgraph_versions,
        diff=record.diff,
        claim_keys=_claim_keys_from_record(record),
        approval=record.approval,
        detail="plan was already committed; returning persisted mutation receipt",
    )


def _errored_commit_result(
    record: GraphMutationPlanRecord,
    *,
    current_subgraph_versions: Mapping[str, int],
    detail: str | None,
    recommended_next_action: str,
) -> GraphMutationCommitResult:
    return GraphMutationCommitResult(
        ok=False,
        plan_id=record.plan_id,
        status=record.status,
        risk=record.risk,
        pot_id=record.pot_id,
        mutation_id=record.mutation_id,
        expected_subgraph_versions=record.expected_subgraph_versions,
        current_subgraph_versions=current_subgraph_versions,
        diff=record.diff,
        claim_keys=_claim_keys_from_record(record),
        approval=record.approval,
        detail=detail,
        recommended_next_action=recommended_next_action,
    )


def _commit_state_changed_result(
    record: GraphMutationPlanRecord,
) -> GraphMutationCommitResult:
    if record.status == GraphMutationPlanStatus.committed.value:
        return _committed_plan_result(record)
    return _concurrent_commit_result(record)


def _concurrent_commit_result(
    record: GraphMutationPlanRecord,
    *,
    detail: str | None = None,
) -> GraphMutationCommitResult:
    if detail is None:
        detail = (
            "plan is already being committed concurrently by another caller"
            if record.status == GraphMutationPlanStatus.committing.value
            else f"plan state changed concurrently to {record.status!r}"
        )
    return GraphMutationCommitResult(
        ok=False,
        plan_id=record.plan_id,
        status=record.status,
        risk=record.risk,
        pot_id=record.pot_id,
        mutation_id=record.mutation_id,
        applied_at=record.committed_at,
        expected_subgraph_versions=record.expected_subgraph_versions,
        current_subgraph_versions=record.current_subgraph_versions,
        new_subgraph_versions=record.final_subgraph_versions,
        diff=record.diff,
        claim_keys=_claim_keys_from_record(record),
        approval=record.approval,
        detail=detail,
        recommended_next_action=(
            "Inspect the stored plan state. Recover a stale committing attempt "
            "with recover_stale=True after the configured claim timeout."
        ),
    )


def _commit_attempt_is_stale(
    record: GraphMutationPlanRecord,
    *,
    now: datetime,
    timeout_seconds: int,
) -> bool:
    started_at = record.commit_attempt_started_at
    if started_at is None:
        return True
    try:
        return now - started_at >= timedelta(seconds=timeout_seconds)
    except TypeError:
        return True


def _claim_keys_from_record(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    keys: list[str] = []
    for op in record.accepted_ops:
        raw = op.get("claim_keys") if isinstance(op, Mapping) else None
        if isinstance(raw, (list, tuple)):
            keys.extend(str(key) for key in raw if key)
    return tuple(dict.fromkeys(keys))


def _mapping_of_ints(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _approval_error(
    record: GraphMutationPlanRecord,
    *,
    approved_by: str | None,
    require_approval: bool = True,
) -> str | None:
    if record.review_required_ops:
        return (
            "plan contains review-required operations that the local commit path "
            "does not apply yet"
        )
    if require_approval and (
        record.status == GraphMutationPlanStatus.review_required.value
        or record.risk in {MutationRisk.medium.value, MutationRisk.high.value}
    ):
        if not (record.approval or (approved_by and approved_by.strip())):
            return "approval required for medium- or high-risk plan"
    return None


def _provenance_for_commit(
    provenance: ProvenanceContext | None,
    *,
    approved_by: str | None,
    mutation_id: str,
    prepared: PreparedMutation | None = None,
) -> ProvenanceContext:
    if provenance is None:
        return ProvenanceContext(
            actor_user_id=approved_by,
            mutation_id=mutation_id,
            prepared=prepared,
        )
    return replace(
        provenance,
        actor_user_id=provenance.actor_user_id or approved_by,
        mutation_id=mutation_id,
        prepared=prepared,
    )


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _safe_int(raw) for key, raw in value.items()}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _unsupported_from_exception(
    exc: CapabilityNotImplemented,
    *,
    fallback: str,
) -> Mapping[str, Any]:
    return {
        "name": getattr(exc, "capability", fallback) or fallback,
        "reason": "not_implemented",
        "detail": getattr(exc, "detail", str(exc)),
    }


def _row_claim_key(row: ClaimRow) -> str:
    return row.claim_key or (
        f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.mutation_id or ''}"
    )


def _dedupe_claim_rows(rows: list[ClaimRow]) -> tuple[ClaimRow, ...]:
    out: list[ClaimRow] = []
    seen: set[tuple] = set()
    for row in rows:
        key = (
            row.record_id,
            row.claim_key,
            row.predicate,
            row.subject_key,
            row.object_key,
            row.mutation_id,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return tuple(out)
