"""Validate, lower and reserve one immutable mutation intent."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from potpie_context_engine.core.graph_contract import (
    MutationRisk,
)
from potpie_context_engine.core.graph_plans import (
    GraphMutationApproval,
    GraphMutationDiff,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
    GraphMutationProposal,
)
from potpie_context_engine.core.semantic_mutations import (
    SemanticMutationParseError,
    SemanticMutationRequest,
    SemanticOperationResolver,
)

from .claim_preparation import prepare_claim_assertions
from .reconciliation_validation import validate_reconciliation_plan
from .errors import MutationBatchValidationError
from .claim_lifecycle import ClaimLifecycleConflict
from .semantic_mutations import SemanticMutationValidationIssue
from .mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
    request_fingerprint as _request_fingerprint,
)

from .mutation_state import (
    _CONFLICT_NEXT_ACTION,
    _commit_next_action,
    _conflict_message,
    _expected_versions,
    _lowered_operation_summary,
    _plan_for_idempotency_key,
    _proposal_from_record,
    _rejected_operation_summaries,
    _status_for_proposal,
    _subgraph_versions,
    _version_conflict,
)
from .mutation_workflow import MutationWorkflow


def propose_mutation(
    self: MutationWorkflow,
    payload: Mapping[str, Any],
    *,
    pot_id: str,
    ttl_seconds: int | None = None,
    approved_by: str | None = None,
    resolve_operations: SemanticOperationResolver | None = None,
) -> GraphMutationProposal:
    """Validate, lower, diff, and persist a mutation plan without writing.

    ``approved_by`` pre-approves the plan: a medium-risk batch validates as
    ``validated`` rather than ``review_required`` and the approval is stored
    on the record, so the commit that follows needs no approval of its own.
    Without it every medium-risk write was three calls — propose, a commit
    refused for want of approval, and the commit again with the flag.
    """
    now = datetime.now(timezone.utc)
    plan_id = f"mutation-plan:{uuid.uuid4().hex[:12]}"
    mutation_id = uuid.uuid4().hex
    approver = (approved_by or "").strip() or None
    current_versions = _subgraph_versions(self.backend, pot_id)
    expected_versions = _expected_versions(payload, current_versions)
    conflict = _version_conflict(expected_versions, current_versions)
    expires_at = now + timedelta(
        seconds=max(1, int(ttl_seconds or self.default_plan_ttl_seconds))
    )
    payload_with_pot = deepcopy(dict(payload))
    payload_with_pot["pot_id"] = pot_id

    idempotency_key = str(payload.get("idempotency_key") or "").strip()
    if idempotency_key:
        matching = _plan_for_idempotency_key(
            self.plan_store, pot_id=pot_id, idempotency_key=idempotency_key
        )
        if matching is not None:
            if _request_fingerprint(matching.original_payload) == _request_fingerprint(
                payload_with_pot
            ):
                return _proposal_from_record(
                    matching,
                    ok=matching.status
                    not in {"invalid", "conflict", "error", "expired", "abandoned"},
                    recommended_next_action=(
                        "Reuse the existing plan_id; conflict or expiry requires a new idempotency_key."
                    ),
                )
            else:
                issue = {
                    "code": "idempotency_key_reused",
                    "message": (
                        f"idempotency_key {idempotency_key!r} is already bound to "
                        f"plan {matching.plan_id!r} with different content"
                    ),
                    "severity": "error",
                    "op_index": None,
                }
                record = GraphMutationPlanRecord(
                    plan_id=plan_id,
                    pot_id=pot_id,
                    status=GraphMutationPlanStatus.invalid.value,
                    risk=MutationRisk.low.value,
                    created_at=now,
                    expires_at=expires_at,
                    original_payload=payload_with_pot,
                    validation_issues=(issue,),
                    rejected_ops=(issue,),
                    expected_subgraph_versions=expected_versions,
                    current_subgraph_versions=current_versions,
                    detail=issue["message"],
                )
                self.plan_store.save(record)
                return _proposal_from_record(
                    record,
                    ok=False,
                    recommended_next_action="Use a new idempotency_key for different content.",
                )

    try:
        request = SemanticMutationRequest.parse(
            (
                {**payload, "operations": [dict(op.raw) for op in resolve_operations()]}
                if resolve_operations is not None
                else payload
            ),
            pot_id=pot_id,
            allow_review_required=approver is not None,
            approved_by=approver,
        )
    except SemanticMutationParseError as exc:
        issue = {
            "code": "invalid_mutation_payload",
            "message": str(exc),
            "severity": "error",
            "op_index": None,
        }
        record = GraphMutationPlanRecord(
            plan_id=plan_id,
            pot_id=pot_id,
            status=GraphMutationPlanStatus.invalid.value,
            risk=MutationRisk.low.value,
            created_at=now,
            expires_at=expires_at,
            original_payload=payload_with_pot,
            validation_issues=(issue,),
            rejected_ops=(issue,),
            expected_subgraph_versions=expected_versions,
            current_subgraph_versions=current_versions,
            detail=str(exc),
        )
        record, _ = _persist_preparation(self, record, idempotency_key)
        return _proposal_from_record(
            record,
            ok=False,
            recommended_next_action="Fix the mutation JSON and use a new idempotency_key.",
        )

    semantic_plan = self.validator(request)
    status = _status_for_proposal(semantic_plan, conflict=bool(conflict))
    if (
        status == GraphMutationPlanStatus.validated.value
        and not self.policy.allow_auto_commit
    ):
        status = GraphMutationPlanStatus.review_required.value
    if status not in {
        GraphMutationPlanStatus.invalid.value,
        GraphMutationPlanStatus.conflict.value,
    }:
        try:
            semantic_plan.prepared_at = now
            self.lowerer(request, semantic_plan)
            validate_reconciliation_plan(
                semantic_plan.batch,
                pot_id,
                definition=self.definition,
                anchor=now,
                provenance=semantic_plan.provenance,
            )
            prepare_claim_assertions(
                semantic_plan.batch,
                pot_id=pot_id,
                mutation_id=mutation_id,
                anchor=now,
                query=self.backend.claim_query,
                definition=self.definition,
            )
        except (ClaimLifecycleConflict, MutationBatchValidationError) as exc:
            status = GraphMutationPlanStatus.invalid.value
            semantic_plan.issues = (
                *semantic_plan.issues,
                SemanticMutationValidationIssue(
                    code="claim_lifecycle_conflict",
                    message=str(exc),
                ),
            )
            semantic_plan.batch = None
        # The entire validator/lowerer read set must have observed one revision.
        after_versions = _subgraph_versions(self.backend, pot_id)
        conflict = _version_conflict(current_versions, after_versions)
        if conflict:
            status = GraphMutationPlanStatus.conflict.value
        if semantic_plan.batch is not None:
            semantic_plan.batch = deepcopy(semantic_plan.batch)

    claim_keys = tuple(
        key for op in semantic_plan.accepted_ops for key in op.claim_keys
    )
    diff = GraphMutationDiff.from_batch(
        semantic_plan.batch,
        claim_keys=claim_keys,
    )
    issues = tuple(issue.to_dict() for issue in semantic_plan.issues)
    rejected = _rejected_operation_summaries(semantic_plan.issues)
    warnings = tuple(
        issue.message for issue in semantic_plan.issues if not issue.is_error
    ) + tuple(semantic_plan.warnings)
    approval = (
        GraphMutationApproval(approved_by=approver, approved_at=now)
        if approver is not None
        and status
        in {
            GraphMutationPlanStatus.validated.value,
            GraphMutationPlanStatus.review_required.value,
        }
        else None
    )
    detail = None
    recommended = None
    if conflict:
        detail = _conflict_message(conflict)
        recommended = _CONFLICT_NEXT_ACTION
    elif status == GraphMutationPlanStatus.invalid.value:
        detail = "; ".join(i.message for i in semantic_plan.errors) or None
        recommended = "Fix the validation errors and run graph propose again."
    elif status == GraphMutationPlanStatus.review_required.value:
        recommended = (
            "Review the plan; it holds operations this host cannot apply yet."
            if semantic_plan.review_required_ops
            else _commit_next_action(plan_id, approved=approval is not None)
        )
    elif status == GraphMutationPlanStatus.validated.value:
        recommended = _commit_next_action(plan_id, approved=True)

    prepared = (
        PreparedMutation(
            pot_id=pot_id,
            mutation_id=mutation_id,
            request_fingerprint=_request_fingerprint(payload_with_pot),
            batch_fingerprint=batch_fingerprint(semantic_plan.batch),
            expected_version=expected_versions["_global"],
            prepared_at=now,
            target_claim_record_ids=tuple(
                sorted(
                    {
                        closure.target.record_id
                        for op in (
                            *semantic_plan.batch.edge_deletes,
                            *semantic_plan.batch.invalidations,
                        )
                        for closure in (op.closures or ())
                    }
                )
            ),
        )
        if semantic_plan.batch is not None
        else None
    )
    record = GraphMutationPlanRecord(
        mutation_id=mutation_id,
        prepared=prepared,
        plan_id=plan_id,
        pot_id=pot_id,
        status=status,
        risk=semantic_plan.risk,
        created_at=now,
        expires_at=expires_at,
        original_payload=payload_with_pot,
        validation_issues=issues,
        accepted_ops=tuple(
            _lowered_operation_summary(op) for op in semantic_plan.accepted_ops
        ),
        review_required_ops=tuple(
            _lowered_operation_summary(op) for op in semantic_plan.review_required_ops
        ),
        rejected_ops=rejected,
        lowered_batch=semantic_plan.batch,
        provenance=semantic_plan.provenance,
        expected_subgraph_versions=expected_versions,
        current_subgraph_versions=current_versions,
        diff=diff,
        warnings=warnings,
        approval=approval,
        detail=detail,
    )
    try:
        stored, inserted = _persist_preparation(self, record, idempotency_key)
    except ValueError as exc:
        return _proposal_from_record(
            replace(
                record, status=GraphMutationPlanStatus.invalid.value, detail=str(exc)
            ),
            ok=False,
            recommended_next_action="Use a new idempotency_key for different content.",
        )
    return _proposal_from_record(
        stored,
        ok=stored.status
        in {
            GraphMutationPlanStatus.validated.value,
            GraphMutationPlanStatus.review_required.value,
        },
        recommended_next_action=recommended
        if inserted
        else "Reuse the existing plan_id; conflict or expiry requires a new idempotency_key.",
    )


def _persist_preparation(
    self: MutationWorkflow, record: GraphMutationPlanRecord, key: str
) -> tuple[GraphMutationPlanRecord, bool]:
    if not key:
        self.plan_store.save(record)
        return record, True
    reserve = getattr(self.plan_store, "reserve_idempotency", None)
    if not callable(reserve):
        raise ValueError("plan store does not support atomic request reservation")
    return reserve(
        record=record,
        idempotency_key=key,
        request_fingerprint=_request_fingerprint(record.original_payload),
    )
