"""Translate direct semantic writes onto the canonical prepared workflow."""

from __future__ import annotations

from dataclasses import asdict

from potpie_context_engine.core.semantic_mutations import (
    SemanticMutationRequest,
    SemanticMutationResult,
    SemanticMutationValidationIssue,
)
from potpie_context_engine.core.workbench_service import GraphWorkbenchService


def mutate_prepared(
    workbench: GraphWorkbenchService, request: SemanticMutationRequest
) -> SemanticMutationResult:
    proposal = workbench.propose(
        {
            "pot_id": request.pot_id,
            "graph_contract_version": request.graph_contract_version,
            "idempotency_key": request.idempotency_key,
            "created_by": asdict(request.created_by),
            "operations": [dict(op.raw) for op in request.operations],
        },
        pot_id=request.pot_id,
        approved_by=request.approved_by,
    )
    record = workbench.plan_store.get(pot_id=request.pot_id, plan_id=proposal.plan_id)
    # A rejected competing reservation may be returned without being persisted.
    accepted = record.accepted_ops if record is not None else ()
    issues = tuple(
        SemanticMutationValidationIssue(
            code=str(issue["code"]),
            message=str(issue["message"]),
            severity=str(issue.get("severity", "error")),
            op_index=issue.get("op_index"),
        )
        for issue in proposal.issues
    )
    common = dict(
        risk=proposal.risk,
        pot_id=request.pot_id,
        plan_id=proposal.plan_id,
        operations_accepted=len(accepted),
        claim_keys=proposal.claim_keys,
        subgraphs=tuple(
            sorted({str(op["subgraph"]) for op in accepted if op.get("subgraph")})
        ),
        warnings=proposal.warnings,
        issues=issues,
    )
    counts = {
        "entity_upserts": proposal.diff.entity_upserts,
        "edge_upserts": proposal.diff.edge_upserts,
        "edge_deletes": proposal.diff.edge_deletes,
        "invalidations": proposal.diff.invalidations,
    }
    if not proposal.ok:
        return SemanticMutationResult(
            ok=False, status="rejected", detail=proposal.detail, **common
        )
    if request.dry_run:
        return SemanticMutationResult(
            ok=True,
            status="validated",
            would_apply=proposal.status == "validated",
            preview=counts,
            **common,
        )
    if proposal.status == "review_required" and not request.approved_by:
        return SemanticMutationResult(
            ok=True,
            status="review_required",
            detail=proposal.recommended_next_action,
            **common,
        )
    result = workbench.commit(
        proposal.plan_id, pot_id=request.pot_id, approved_by=request.approved_by
    )
    return SemanticMutationResult(
        ok=result.ok,
        status="applied" if result.ok else result.status,
        auto_committed=result.ok,
        mutation_id=result.mutation_id,
        operations_applied=len(accepted) if result.ok else 0,
        mutations_applied=counts if result.ok else {},
        detail=result.detail,
        **common,
    )
