"""Single coordinator for execution ownership, atomic apply and recovery."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from potpie_context_engine.core.graph_contract import (
    MutationRisk,
)
from potpie_context_engine.core.graph_plans import (
    TERMINAL_PLAN_STATUSES,
    GraphIngestionVerificationResult,
    GraphMutationApproval,
    GraphMutationCommitResult,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
)
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)

from .mutation_state import (
    _CONFLICT_NEXT_ACTION,
    _apply_if_current,
    _approval_error,
    _batch_has_work,
    _claim_keys_from_record,
    _commit_attempt_is_stale,
    _commit_next_action,
    _commit_state_changed_result,
    _commits_landed_since,
    _committed_plan_result,
    _concurrent_commit_result,
    _conflict_message,
    _errored_commit_result,
    _lookup_mutation_execution,
    _provenance_for_commit,
    _subgraph_versions,
    _version_conflict,
)
from .mutation_verification import (
    _quality_count_map,
    _verification_next_action,
)
from .mutation_workflow import MutationWorkflow

_LOG = logging.getLogger(__name__)


def commit_mutation(
    self: MutationWorkflow,
    plan_id: str,
    *,
    pot_id: str,
    approved_by: str | None = None,
    verify: bool = False,
    recover_stale: bool = False,
    defer_verification: bool = False,
    verify_receipt: Callable[..., GraphIngestionVerificationResult],
    quality_snapshot: Callable[..., Mapping[str, Any]],
) -> GraphMutationCommitResult:
    """Apply a plan; defer_verification returns the receipt before readback.

    With verify=True the quality baseline is persisted for verify_commit,
    even when post-commit checks are deferred to a separate RPC.
    """
    now = datetime.now(timezone.utc)
    record = self.plan_store.get(pot_id=pot_id, plan_id=plan_id)
    if record is None:
        return GraphMutationCommitResult(
            ok=False,
            plan_id=plan_id,
            status="not_found",
            risk=MutationRisk.low.value,
            pot_id=pot_id,
            detail=f"mutation plan {plan_id!r} was not found for this pot",
            recommended_next_action="Run graph propose and commit the returned plan_id.",
        )

    record.validate_prepared()

    if record.status == GraphMutationPlanStatus.committed.value:
        receipt = _committed_plan_result(record)
        if verify and not defer_verification:
            receipt = replace(
                receipt, verification=self.verify_commit(plan_id, pot_id=pot_id)
            )
        return receipt

    if record.status in TERMINAL_PLAN_STATUSES:
        return GraphMutationCommitResult(
            ok=False,
            plan_id=record.plan_id,
            status=record.status,
            risk=record.risk,
            pot_id=record.pot_id,
            expected_subgraph_versions=record.expected_subgraph_versions,
            current_subgraph_versions=record.current_subgraph_versions,
            diff=record.diff,
            claim_keys=_claim_keys_from_record(record),
            approval=record.approval,
            detail=f"plan is {record.status} and cannot be committed",
            recommended_next_action="Create a fresh proposal if a write is still needed.",
        )
    retrying_execution = (
        record.status == GraphMutationPlanStatus.error.value
        and record.mutation_id is not None
    )
    recovering_stale = False
    if record.status == GraphMutationPlanStatus.committing.value:
        if not recover_stale:
            return _concurrent_commit_result(record)
        if not _commit_attempt_is_stale(
            record,
            now=now,
            timeout_seconds=self.commit_claim_timeout_seconds,
        ):
            return _concurrent_commit_result(
                record,
                detail=("commit attempt is still active and cannot be recovered yet"),
            )
        recovering_stale = True

    if retrying_execution or recovering_stale:
        lookup = _lookup_mutation_execution(self.backend, record)
        if lookup.state == MutationExecutionState.completed.value:
            return _complete_recovered_commit(
                self,
                record,
                verify_receipt=verify_receipt,
                lookup=lookup,
                now=now,
                verify=verify and not defer_verification,
            )
        if lookup.state == MutationExecutionState.in_flight.value:
            return _concurrent_commit_result(
                record,
                detail="the backend mutation is still in flight",
            )
        if lookup.state != MutationExecutionState.absent.value:
            return _fail_unsafe_recovery(self, record, lookup=lookup)

    if record.is_expired(now=now):
        expired = replace(
            record,
            status=GraphMutationPlanStatus.expired.value,
            detail="plan expired before commit",
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=expired,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
            )
        return GraphMutationCommitResult(
            ok=False,
            plan_id=expired.plan_id,
            status=expired.status,
            risk=expired.risk,
            pot_id=expired.pot_id,
            expected_subgraph_versions=expired.expected_subgraph_versions,
            current_subgraph_versions=_subgraph_versions(self.backend, pot_id),
            diff=expired.diff,
            claim_keys=_claim_keys_from_record(expired),
            detail="plan expired before commit",
            recommended_next_action="Reread current graph state and propose a new plan.",
        )

    current_versions = _subgraph_versions(self.backend, pot_id)
    conflict = _version_conflict(
        record.expected_subgraph_versions,
        current_versions,
    )
    if conflict:
        landed, landed_count = _commits_landed_since(
            self.plan_store,
            pot_id=pot_id,
            since=record.created_at,
            exclude_plan_id=record.plan_id,
        )
        conflicted = replace(
            record,
            status=GraphMutationPlanStatus.conflict.value,
            current_subgraph_versions=current_versions,
            detail=_conflict_message(
                conflict, landed=landed, landed_count=landed_count
            ),
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=conflicted,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
            )
        return GraphMutationCommitResult(
            ok=False,
            plan_id=conflicted.plan_id,
            status=conflicted.status,
            risk=conflicted.risk,
            pot_id=conflicted.pot_id,
            expected_subgraph_versions=conflicted.expected_subgraph_versions,
            current_subgraph_versions=current_versions,
            diff=conflicted.diff,
            claim_keys=_claim_keys_from_record(conflicted),
            detail=conflicted.detail,
            recommended_next_action=_CONFLICT_NEXT_ACTION,
        )

    approval = record.approval
    approval_error = _approval_error(
        record,
        approved_by=approved_by,
        require_approval=self.policy.require_approval_for_review,
    )
    if approval_error:
        return GraphMutationCommitResult(
            ok=False,
            plan_id=record.plan_id,
            status=record.status,
            risk=record.risk,
            pot_id=record.pot_id,
            expected_subgraph_versions=record.expected_subgraph_versions,
            current_subgraph_versions=current_versions,
            diff=record.diff,
            claim_keys=_claim_keys_from_record(record),
            approval=approval,
            detail=approval_error,
            recommended_next_action=_commit_next_action(plan_id, approved=False),
        )
    if approved_by and approval is None:
        approval = GraphMutationApproval(
            approved_by=approved_by,
            approved_at=now,
        )

    quality_before = (
        quality_snapshot(
            self.backend,
            pot_id=pot_id,
            definition=self.definition,
        )
        if verify
        else None
    )
    execution_mutation_id = record.mutation_id or uuid.uuid4().hex
    attempt_started_at = datetime.now(timezone.utc)
    reserved = replace(
        record,
        status=GraphMutationPlanStatus.committing.value,
        approval=approval,
        mutation_id=execution_mutation_id,
        commit_attempt_id=uuid.uuid4().hex,
        commit_attempt_started_at=attempt_started_at,
        verification_quality_before={
            "status": quality_before["status"],
            "quality_counts": _quality_count_map(quality_before),
        }
        if quality_before is not None
        else None,
        detail=None,
    )
    if not self.plan_store.compare_and_set(
        expected=record,
        replacement=reserved,
    ):
        return _commit_state_changed_result(
            self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
        )
    record = reserved

    if record.lowered_batch is None or not _batch_has_work(record.lowered_batch):
        committed = replace(
            record,
            status=GraphMutationPlanStatus.committed.value,
            commit_attempt_id=None,
            commit_attempt_started_at=None,
            committed_at=now,
            final_subgraph_versions=current_versions,
            approval=approval,
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=committed,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
            )
        verification = (
            verify_receipt(
                self.backend,
                pot_id=pot_id,
                record=committed,
                before_quality=quality_before,
                definition=self.definition,
            )
            if verify and not defer_verification
            else None
        )
        return GraphMutationCommitResult(
            ok=True,
            plan_id=committed.plan_id,
            status=committed.status,
            risk=committed.risk,
            pot_id=committed.pot_id,
            mutation_id=execution_mutation_id,
            applied_at=now,
            expected_subgraph_versions=committed.expected_subgraph_versions,
            current_subgraph_versions=current_versions,
            new_subgraph_versions=current_versions,
            diff=committed.diff,
            claim_keys=_claim_keys_from_record(committed),
            approval=approval,
            verification=verification,
            detail="plan had no structural mutations to apply",
            recommended_next_action=_verification_next_action(verification),
        )

    commit_actor = approved_by or (
        approval.approved_by if approval is not None else None
    )
    compare_and_apply = getattr(self.backend.mutation, "compare_and_apply", None)
    atomic_apply: Callable[[int], Any] | None = None
    if callable(compare_and_apply):

        def atomic_apply(expected_version: int) -> Any:
            return compare_and_apply(
                deepcopy(record.lowered_batch),
                expected_pot_id=pot_id,
                expected_version=expected_version,
                provenance_context=_provenance_for_commit(
                    record.provenance,
                    approved_by=commit_actor,
                    mutation_id=execution_mutation_id,
                    prepared=record.prepared,
                ),
                reconciliation_config=self.reconciliation_config,
            )

    try:
        result, apply_versions = _apply_if_current(
            self.backend,
            pot_id=pot_id,
            expected_versions=record.expected_subgraph_versions,
            apply=lambda: self.backend.mutation.apply(
                deepcopy(record.lowered_batch),
                expected_pot_id=pot_id,
                provenance_context=_provenance_for_commit(
                    record.provenance,
                    approved_by=commit_actor,
                    mutation_id=execution_mutation_id,
                    prepared=record.prepared,
                ),
            ),
            atomic_apply=atomic_apply,
            reconciliation_config=self.reconciliation_config,
        )
        if result is None:
            conflict = _version_conflict(
                record.expected_subgraph_versions, apply_versions
            )
            conflicted = replace(
                record,
                status=GraphMutationPlanStatus.conflict.value,
                current_subgraph_versions=apply_versions,
                commit_attempt_id=None,
                commit_attempt_started_at=None,
                detail=_conflict_message(conflict or {}),
            )
            self.plan_store.compare_and_set(expected=record, replacement=conflicted)
            return GraphMutationCommitResult(
                ok=False,
                plan_id=plan_id,
                status=GraphMutationPlanStatus.conflict.value,
                risk=record.risk,
                pot_id=pot_id,
                mutation_id=execution_mutation_id,
                expected_subgraph_versions=record.expected_subgraph_versions,
                current_subgraph_versions=apply_versions,
                diff=record.diff,
                claim_keys=_claim_keys_from_record(record),
                detail=conflicted.detail,
                recommended_next_action=_CONFLICT_NEXT_ACTION,
            )
        final_versions = _subgraph_versions(self.backend, pot_id) if result.ok else None
    except Exception:
        _LOG.exception(
            "graph backend mutation failed unexpectedly "
            "(pot_id=%s, plan_id=%s, mutation_id=%s)",
            pot_id,
            plan_id,
            execution_mutation_id,
        )
        detail = "backend_exception: backend mutation failed unexpectedly"
        errored = replace(
            record,
            status=GraphMutationPlanStatus.error.value,
            current_subgraph_versions=current_versions,
            commit_attempt_id=None,
            commit_attempt_started_at=None,
            approval=approval,
            detail=detail,
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=errored,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
            )
        return _errored_commit_result(
            errored,
            current_subgraph_versions=current_versions,
            detail=detail,
            recommended_next_action=(
                "Retry this plan_id. Recovery will replay only when the "
                "backend can prove the mutation is absent or completed. If "
                "a worker crashed, wait for the claim timeout and retry with "
                "recover_stale=True."
            ),
        )

    if not result.ok:
        errored = replace(
            record,
            status=GraphMutationPlanStatus.error.value,
            current_subgraph_versions=current_versions,
            commit_attempt_id=None,
            commit_attempt_started_at=None,
            approval=approval,
            detail=result.error,
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=errored,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
            )
        return _errored_commit_result(
            errored,
            current_subgraph_versions=current_versions,
            detail=result.error,
            recommended_next_action="Inspect backend readiness with `potpie graph status --json`.",
        )

    assert final_versions is not None
    committed = replace(
        record,
        status=GraphMutationPlanStatus.committed.value,
        mutation_id=execution_mutation_id,
        commit_attempt_id=None,
        commit_attempt_started_at=None,
        committed_at=now,
        final_subgraph_versions=final_versions,
        approval=approval,
    )
    if not self.plan_store.compare_and_set(
        expected=record,
        replacement=committed,
    ):
        current = self.plan_store.get(pot_id=pot_id, plan_id=plan_id) or record
        return GraphMutationCommitResult(
            ok=False,
            plan_id=plan_id,
            status=current.status,
            risk=current.risk,
            pot_id=pot_id,
            mutation_id=execution_mutation_id,
            applied_at=now,
            expected_subgraph_versions=current.expected_subgraph_versions,
            current_subgraph_versions=final_versions,
            new_subgraph_versions=final_versions,
            diff=current.diff,
            claim_keys=_claim_keys_from_record(current),
            approval=approval,
            detail=(
                "backend mutation completed, but the plan store rejected the "
                "final compare-and-set; the stable mutation_id is recoverable"
            ),
            recommended_next_action=(
                "Inspect the stored plan. If it remains committing after the "
                "claim timeout, retry with recover_stale=True."
            ),
        )
    verification = (
        verify_receipt(
            self.backend,
            pot_id=pot_id,
            record=committed,
            before_quality=quality_before,
            definition=self.definition,
        )
        if verify and not defer_verification
        else None
    )
    return GraphMutationCommitResult(
        ok=True,
        plan_id=committed.plan_id,
        status=committed.status,
        risk=committed.risk,
        pot_id=committed.pot_id,
        mutation_id=execution_mutation_id,
        applied_at=now,
        expected_subgraph_versions=committed.expected_subgraph_versions,
        current_subgraph_versions=current_versions,
        new_subgraph_versions=final_versions,
        diff=committed.diff,
        claim_keys=_claim_keys_from_record(committed),
        approval=approval,
        verification=verification,
        recommended_next_action=_verification_next_action(verification),
    )


def _complete_recovered_commit(
    self: MutationWorkflow,
    record: GraphMutationPlanRecord,
    *,
    lookup: MutationExecutionLookup,
    now: datetime,
    verify: bool,
    verify_receipt: Callable[..., GraphIngestionVerificationResult],
) -> GraphMutationCommitResult:
    """Finalize a plan from a backend-proven completed execution receipt."""

    result = lookup.result
    if (
        result is None
        or not record.mutation_id
        or result.mutation_id != record.mutation_id
    ):
        invalid_lookup = replace(
            lookup,
            state=MutationExecutionState.ambiguous.value,
            detail=None,
        )
        return _fail_unsafe_recovery(self, record, lookup=invalid_lookup)

    current_versions = _subgraph_versions(self.backend, record.pot_id)
    if not result.ok:
        errored = replace(
            record,
            status=GraphMutationPlanStatus.error.value,
            current_subgraph_versions=current_versions,
            commit_attempt_id=None,
            commit_attempt_started_at=None,
            detail=result.error,
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=errored,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(
                    pot_id=record.pot_id,
                    plan_id=record.plan_id,
                )
                or record
            )
        return _errored_commit_result(
            errored,
            current_subgraph_versions=current_versions,
            detail=result.error,
            recommended_next_action=(
                "Inspect backend readiness before creating a fresh proposal."
            ),
        )

    committed = replace(
        record,
        status=GraphMutationPlanStatus.committed.value,
        mutation_id=result.mutation_id,
        commit_attempt_id=None,
        commit_attempt_started_at=None,
        committed_at=now,
        final_subgraph_versions=current_versions,
        detail=None,
    )
    if not self.plan_store.compare_and_set(
        expected=record,
        replacement=committed,
    ):
        return _commit_state_changed_result(
            self.plan_store.get(
                pot_id=record.pot_id,
                plan_id=record.plan_id,
            )
            or record
        )
    verification = (
        verify_receipt(
            self.backend,
            pot_id=record.pot_id,
            record=committed,
            before_quality=record.verification_quality_before,
            definition=self.definition,
        )
        if verify
        else None
    )
    return GraphMutationCommitResult(
        ok=True,
        plan_id=committed.plan_id,
        status=committed.status,
        risk=committed.risk,
        pot_id=committed.pot_id,
        mutation_id=committed.mutation_id,
        applied_at=committed.committed_at,
        expected_subgraph_versions=committed.expected_subgraph_versions,
        current_subgraph_versions=current_versions,
        new_subgraph_versions=current_versions,
        diff=committed.diff,
        claim_keys=_claim_keys_from_record(committed),
        approval=committed.approval,
        verification=verification,
        detail="recovered the backend's persisted mutation receipt",
        recommended_next_action=_verification_next_action(verification),
    )


def _fail_unsafe_recovery(
    self: MutationWorkflow,
    record: GraphMutationPlanRecord,
    *,
    lookup: MutationExecutionLookup,
) -> GraphMutationCommitResult:
    """Refuse to execute again when the backend cannot prove it is safe."""

    state = str(lookup.state)
    detail = (
        "mutation_recovery_unsupported: the backend cannot prove whether "
        "the reserved mutation executed"
        if state == MutationExecutionState.unsupported.value
        else "mutation_recovery_ambiguous: backend execution state is uncertain"
    )
    current = record
    if record.status == GraphMutationPlanStatus.committing.value:
        errored = replace(
            record,
            status=GraphMutationPlanStatus.error.value,
            commit_attempt_id=None,
            commit_attempt_started_at=None,
            detail=detail,
        )
        if not self.plan_store.compare_and_set(
            expected=record,
            replacement=errored,
        ):
            return _commit_state_changed_result(
                self.plan_store.get(
                    pot_id=record.pot_id,
                    plan_id=record.plan_id,
                )
                or record
            )
        current = errored
    return _errored_commit_result(
        current,
        current_subgraph_versions=_subgraph_versions(
            self.backend,
            record.pot_id,
        ),
        detail=detail,
        recommended_next_action=(
            "Inspect graph history using the mutation_id. Create a fresh "
            "proposal only after reconciling whether the mutation landed."
        ),
    )


def commit_status(self, plan_id: str, *, pot_id: str) -> GraphMutationCommitResult:
    """Read a durable plan receipt without touching the graph or applying work."""
    record = self.plan_store.get(pot_id=pot_id, plan_id=plan_id)
    if record is None:
        return GraphMutationCommitResult(
            ok=False,
            plan_id=plan_id,
            pot_id=pot_id,
            risk="low",
            status="not_found",
        )
    if record.status == GraphMutationPlanStatus.committed.value:
        return _committed_plan_result(record)
    if record.status == GraphMutationPlanStatus.committing.value:
        return _concurrent_commit_result(record)
    if record.status == GraphMutationPlanStatus.error.value:
        next_action = (
            "Inspect graph history for this plan and backend readiness before retrying."
        )
    elif record.status in TERMINAL_PLAN_STATUSES:
        next_action = "Create a fresh proposal if a write is still needed."
    else:
        next_action = _commit_next_action(plan_id, approved=record.approval is not None)
    return replace(
        _committed_plan_result(record),
        ok=False,
        detail=record.detail or f"plan is {record.status}",
        recommended_next_action=next_action,
    )
