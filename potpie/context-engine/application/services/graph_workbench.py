"""Shared Graph V2 workbench envelope assembly."""

from __future__ import annotations

import uuid
from dataclasses import replace
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from application.services.semantic_mutation_lowering import lower_semantic_request
from application.services.semantic_mutation_validator import validate_semantic_request
from domain.graph_contract import GRAPH_CONTRACT_VERSION as DATA_PLANE_CONTRACT_VERSION
from domain.graph_contract import MutationRisk
from domain.graph_mutations import ProvenanceContext
from domain.graph_plans import (
    GraphMutationApproval,
    GraphMutationCommitResult,
    GraphMutationDiff,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
    GraphMutationProposal,
    TERMINAL_PLAN_STATUSES,
)
from domain.graph_workbench import (
    GRAPH_WORKBENCH_ADMIN_COMMANDS,
    GRAPH_WORKBENCH_COMMANDS,
    GRAPH_WORKBENCH_CONTRACT_VERSION,
    GRAPH_WORKBENCH_LEGACY_COMMANDS,
    GraphCommandEnvelope,
    GraphCommandError,
    GraphUnsupported,
    GraphUnsupportedResult,
)
from domain.graph_workbench_ontology import ranked_catalog_views
from domain.ports.graph.backend import GraphBackend
from domain.ports.graph.plan_store import GraphPlanStorePort
from domain.semantic_mutations import (
    LoweredOperation,
    SemanticMutationParseError,
    SemanticMutationRequest,
    SemanticMutationValidationIssue,
)

_DEFAULT_PLAN_TTL_SECONDS = 3600

_CROSS_CUTTING_RESULT_KEYS = frozenset(
    {
        "ok",
        "graph_contract_version",
        "ontology_version",
        "subgraph_versions",
        "pot_id",
        "warnings",
        "unsupported",
        "recommended_next_action",
    }
)


class GraphWorkbenchService:
    """Graph V2 workbench workflow layer above the backend write door."""

    def __init__(
        self,
        *,
        backend: GraphBackend,
        plan_store: GraphPlanStorePort,
        default_plan_ttl_seconds: int = _DEFAULT_PLAN_TTL_SECONDS,
    ) -> None:
        self.backend = backend
        self.plan_store = plan_store
        self.default_plan_ttl_seconds = default_plan_ttl_seconds

    def propose(
        self,
        payload: Mapping[str, Any],
        *,
        pot_id: str,
        ttl_seconds: int | None = None,
    ) -> GraphMutationProposal:
        """Validate, lower, diff, and persist a mutation plan without writing."""
        now = datetime.now(timezone.utc)
        plan_id = f"mutation-plan:{uuid.uuid4().hex[:12]}"
        current_versions = _subgraph_versions(self.backend, pot_id)
        expected_versions = _expected_versions(payload, current_versions)
        conflict = _version_conflict(expected_versions, current_versions)
        expires_at = now + timedelta(
            seconds=max(1, int(ttl_seconds or self.default_plan_ttl_seconds))
        )
        payload_with_pot = dict(payload)
        payload_with_pot["pot_id"] = pot_id

        try:
            request = SemanticMutationRequest.parse(payload, pot_id=pot_id)
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
            self.plan_store.save(record)
            return _proposal_from_record(
                record,
                ok=False,
                recommended_next_action="Fix the mutation JSON and run graph propose again.",
            )

        semantic_plan = validate_semantic_request(request)
        status = _status_for_proposal(semantic_plan, conflict=bool(conflict))
        if status not in {
            GraphMutationPlanStatus.invalid.value,
            GraphMutationPlanStatus.conflict.value,
        }:
            lower_semantic_request(request, semantic_plan)

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
        detail = None
        recommended = None
        if conflict:
            detail = _conflict_message(conflict)
            recommended = "Reread the affected graph views and propose a new plan."
        elif status == GraphMutationPlanStatus.invalid.value:
            detail = "; ".join(i.message for i in semantic_plan.errors) or None
            recommended = "Fix the validation errors and run graph propose again."
        elif status == GraphMutationPlanStatus.review_required.value:
            recommended = (
                "Review the persisted plan, then commit with --approved-by when policy allows."
            )
        elif status == GraphMutationPlanStatus.validated.value:
            recommended = f"Commit with `potpie graph commit {plan_id} --json`."

        record = GraphMutationPlanRecord(
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
                _lowered_operation_summary(op)
                for op in semantic_plan.review_required_ops
            ),
            rejected_ops=rejected,
            lowered_batch=semantic_plan.batch,
            provenance=semantic_plan.provenance,
            expected_subgraph_versions=expected_versions,
            current_subgraph_versions=current_versions,
            diff=diff,
            warnings=warnings,
            detail=detail,
        )
        self.plan_store.save(record)
        return _proposal_from_record(
            record,
            ok=status
            in {
                GraphMutationPlanStatus.validated.value,
                GraphMutationPlanStatus.review_required.value,
            },
            recommended_next_action=recommended,
        )

    def commit(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
    ) -> GraphMutationCommitResult:
        """Apply an unexpired server-created plan by id."""
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

        if record.is_expired(now=now):
            expired = replace(
                record,
                status=GraphMutationPlanStatus.expired.value,
                detail="plan expired before commit",
            )
            self.plan_store.save(expired)
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
        conflict = _version_conflict(record.expected_subgraph_versions, current_versions)
        if conflict:
            conflicted = replace(
                record,
                status=GraphMutationPlanStatus.conflict.value,
                current_subgraph_versions=current_versions,
                detail=_conflict_message(conflict),
            )
            self.plan_store.save(conflicted)
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
                recommended_next_action="Reread current graph state and propose a new plan.",
            )

        approval = record.approval
        approval_error = _approval_error(record, approved_by=approved_by)
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
                recommended_next_action=(
                    f"Review the plan, then run `potpie graph commit {plan_id} "
                    "--approved-by <user-ref> --json` when policy allows."
                ),
            )
        if approved_by and approval is None:
            approval = GraphMutationApproval(
                approved_by=approved_by,
                approved_at=now,
            )
            record = replace(
                record,
                status=GraphMutationPlanStatus.approved.value,
                approval=approval,
            )
            self.plan_store.save(record)

        if record.lowered_batch is None or not _batch_has_work(record.lowered_batch):
            committed = replace(
                record,
                status=GraphMutationPlanStatus.committed.value,
                committed_at=now,
                final_subgraph_versions=current_versions,
                approval=approval,
            )
            self.plan_store.save(committed)
            return GraphMutationCommitResult(
                ok=True,
                plan_id=committed.plan_id,
                status=committed.status,
                risk=committed.risk,
                pot_id=committed.pot_id,
                applied_at=now,
                expected_subgraph_versions=committed.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                new_subgraph_versions=current_versions,
                diff=committed.diff,
                claim_keys=_claim_keys_from_record(committed),
                approval=approval,
                detail="plan had no structural mutations to apply",
            )

        result = self.backend.mutation.apply(
            record.lowered_batch,
            expected_pot_id=pot_id,
            provenance_context=_provenance_for_commit(
                record.provenance,
                approved_by=approved_by,
            ),
        )
        if not result.ok:
            errored = replace(
                record,
                status=GraphMutationPlanStatus.error.value,
                current_subgraph_versions=current_versions,
                approval=approval,
                detail=result.error,
            )
            self.plan_store.save(errored)
            return GraphMutationCommitResult(
                ok=False,
                plan_id=errored.plan_id,
                status=errored.status,
                risk=errored.risk,
                pot_id=errored.pot_id,
                expected_subgraph_versions=errored.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                diff=errored.diff,
                claim_keys=_claim_keys_from_record(errored),
                approval=approval,
                detail=result.error,
                recommended_next_action="Inspect backend readiness with `potpie graph status --json`.",
            )

        final_versions = _subgraph_versions(self.backend, pot_id)
        committed = replace(
            record,
            status=GraphMutationPlanStatus.committed.value,
            mutation_id=result.mutation_id,
            committed_at=now,
            final_subgraph_versions=final_versions,
            approval=approval,
        )
        self.plan_store.save(committed)
        return GraphMutationCommitResult(
            ok=True,
            plan_id=committed.plan_id,
            status=committed.status,
            risk=committed.risk,
            pot_id=committed.pot_id,
            mutation_id=result.mutation_id,
            applied_at=now,
            expected_subgraph_versions=committed.expected_subgraph_versions,
            current_subgraph_versions=current_versions,
            new_subgraph_versions=final_versions,
            diff=committed.diff,
            claim_keys=_claim_keys_from_record(committed),
            approval=approval,
        )


def new_graph_request_id() -> str:
    return f"req:{uuid.uuid4().hex}"


def graph_success_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    result: Mapping[str, Any] | None = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=True,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=dict(result or {}),
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
    )


def graph_error_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    code: str,
    message: str,
    detail: Any = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=False,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=None,
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
        error=GraphCommandError(code=code, message=message, detail=detail),
    )


def graph_not_implemented_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    detail: str | None = None,
    recommended_next_action: str | None = None,
) -> GraphCommandEnvelope:
    message = f"{command} is not implemented yet"
    return graph_error_envelope(
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        code="not_implemented",
        message=message,
        detail=detail,
        unsupported=(
            GraphUnsupported(
                name=command,
                reason="not_implemented",
                detail=detail,
            ),
        ),
        recommended_next_action=recommended_next_action,
    )


def graph_not_implemented_result(command: str, *, detail: str | None = None) -> dict:
    return GraphUnsupportedResult(
        status="not_implemented",
        command=command,
        detail=detail,
    ).to_dict()


def normalize_workbench_result(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, int], tuple[str, ...], tuple[GraphUnsupported, ...]
]:
    """Move cross-cutting fields from a legacy result into envelope fields."""
    result = dict(payload)
    subgraph_versions = _mapping_of_ints(result.pop("subgraph_versions", {}))
    warnings = _string_tuple(result.pop("warnings", ()))
    unsupported = _unsupported_tuple(result.pop("unsupported", ()))
    unsupported += _unsupported_tuple(result.pop("unsupported_includes", ()))
    for key in _CROSS_CUTTING_RESULT_KEYS:
        result.pop(key, None)
    return result, subgraph_versions, warnings, unsupported


def normalize_catalog_result(
    payload: Mapping[str, Any],
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """Project the V1.5 data-plane catalog as a V2 workbench catalog body."""
    result = dict(payload)
    data_plane_version = result.pop(
        "graph_contract_version", DATA_PLANE_CONTRACT_VERSION
    )
    result.pop("ontology_version", None)
    result.pop("ok", None)
    result["data_plane_graph_contract_version"] = data_plane_version
    result["workbench_graph_contract_version"] = GRAPH_WORKBENCH_CONTRACT_VERSION
    result["commands"] = list(GRAPH_WORKBENCH_COMMANDS)
    result["admin_commands"] = list(GRAPH_WORKBENCH_ADMIN_COMMANDS)
    result["legacy_commands"] = list(GRAPH_WORKBENCH_LEGACY_COMMANDS)
    views, ranking = ranked_catalog_views(
        result.get("views", ()),
        task,
    )
    result["views"] = views
    if task:
        result["task"] = task
        result["task_ranking"] = ranking
    result["transition"] = {
        "legacy_commands_callable": True,
        "mutate_replacement": "propose + commit",
        "inspect_replacement": "neighborhood",
    }
    return result


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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return (str(value),)


def _unsupported_tuple(value: Any) -> tuple[GraphUnsupported, ...]:
    if value is None:
        return ()
    if isinstance(value, GraphUnsupported):
        return (value,)
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return (GraphUnsupported(name=str(value), reason="unsupported"),)
    out: list[GraphUnsupported] = []
    for item in value:
        if isinstance(item, GraphUnsupported):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            out.append(
                GraphUnsupported(
                    name=str(item.get("name") or item.get("include") or "unknown"),
                    reason=str(item.get("reason") or item.get("code") or "unsupported"),
                    detail=item.get("detail"),
                )
            )
            continue
        out.append(GraphUnsupported(name=str(item), reason="unsupported"))
    return tuple(out)


def _subgraph_versions(backend: GraphBackend, pot_id: str) -> dict[str, int]:
    try:
        counts = dict(backend.analytics.counts(pot_id))
    except Exception:
        counts = {}
    try:
        return {"_global": int(counts.get("claims", 0))}
    except (TypeError, ValueError):
        return {"_global": 0}


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


def _conflict_message(conflict: Mapping[str, Any]) -> str:
    return (
        f"{conflict.get('subgraph')} changed after the plan was proposed "
        f"(expected {conflict.get('expected_version')}, "
        f"actual {conflict.get('actual_version')})"
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


def _claim_keys_from_record(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    keys: list[str] = []
    for op in record.accepted_ops:
        raw = op.get("claim_keys") if isinstance(op, Mapping) else None
        if isinstance(raw, (list, tuple)):
            keys.extend(str(key) for key in raw if key)
    return tuple(dict.fromkeys(keys))


def _approval_error(
    record: GraphMutationPlanRecord,
    *,
    approved_by: str | None,
) -> str | None:
    if record.review_required_ops or record.risk == MutationRisk.high.value:
        return (
            "plan contains high-risk or review-required operations that the local "
            "Phase 3 commit path does not apply yet"
        )
    if record.status == GraphMutationPlanStatus.review_required.value or (
        record.risk == MutationRisk.medium.value
    ):
        if not (record.approval or (approved_by and approved_by.strip())):
            return "approval required for medium-risk plan"
    return None


def _batch_has_work(batch) -> bool:
    return bool(
        batch.entity_upserts
        or batch.edge_upserts
        or batch.edge_deletes
        or batch.invalidations
    )


def _provenance_for_commit(
    provenance: ProvenanceContext | None,
    *,
    approved_by: str | None,
) -> ProvenanceContext | None:
    if not approved_by:
        return provenance
    if provenance is None:
        return ProvenanceContext(actor_user_id=approved_by)
    if provenance.actor_user_id:
        return provenance
    return replace(provenance, actor_user_id=approved_by)


__all__ = [
    "GraphWorkbenchService",
    "graph_error_envelope",
    "graph_not_implemented_envelope",
    "graph_not_implemented_result",
    "graph_success_envelope",
    "new_graph_request_id",
    "normalize_catalog_result",
    "normalize_workbench_result",
]
