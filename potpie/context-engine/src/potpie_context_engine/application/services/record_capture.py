"""Persist retryable record requests through the shared graph workbench."""

from __future__ import annotations

from dataclasses import asdict

from potpie_context_core.ports.agent_context import RecordReceipt, RecordRequest
from potpie_context_core.semantic_mutations import SemanticMutationRequest
from potpie_context_core.workbench_service import GraphWorkbenchService


def record_once(
    request: RecordRequest,
    semantic: SemanticMutationRequest,
    *,
    source_id: str,
    workbench: GraphWorkbenchService | None,
) -> RecordReceipt:
    """Reuse the stored lowering and apply receipt, including after a restart.

    A data-plane service constructed alone has no durable plan store. Refuse an
    explicit retry guarantee there instead of silently applying the write again.
    ``build_graph_runtime`` wires the same workbench used by propose/commit.
    """
    if workbench is None:
        raise ValueError(
            "record idempotency requires a plan store; construct the service with "
            "build_graph_runtime"
        )
    payload = {
        "pot_id": request.pot_id,
        "idempotency_key": semantic.idempotency_key,
        "created_by": asdict(semantic.created_by),
        "operations": [dict(op.raw) for op in semantic.operations],
    }
    proposal = workbench.propose(
        payload, pot_id=request.pot_id, approved_by=semantic.approved_by
    )
    replay = proposal.status == "committed"
    result = proposal
    if proposal.ok:
        result = workbench.commit(proposal.plan_id, pot_id=request.pot_id)
    accepted = result.ok and result.status == "committed"
    subgraphs = sorted({op.subgraph for op in semantic.operations if op.subgraph})
    return RecordReceipt(
        pot_id=request.pot_id,
        record_type=request.record_type.strip().lower(),
        accepted=accepted,
        record_id=source_id,
        status=("duplicate" if replay else "recorded") if accepted else "rejected",
        mutations_applied=len(semantic.operations) if accepted and not replay else 0,
        detail=result.detail,
        metadata={
            "plan_id": result.plan_id,
            "mutation_id": getattr(result, "mutation_id", None),
            "graph_contract_version": result.graph_contract_version,
            "ontology_version": result.ontology_version,
            "claim_keys": list(result.claim_keys),
            "subgraph": subgraphs[0] if subgraphs else None,
            "subgraphs": subgraphs,
            "truth": semantic.operations[0].truth,
            "risk": result.risk,
            "auto_committed": accepted,
        },
    )
