"""Commit response examples shared by CLI transport tests."""

from __future__ import annotations

from datetime import datetime, timezone

from potpie_context_engine.core.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationCommitResult,
    GraphMutationDiff,
)


def _commit_result(
    *,
    ok: bool = True,
    status: str = "committed",
    plan_id: str = "mutation-plan:test",
    verification: GraphIngestionVerificationResult | None = None,
) -> GraphMutationCommitResult:
    return GraphMutationCommitResult(
        ok=ok,
        plan_id=plan_id,
        status=status,
        risk="low",
        pot_id="p",
        mutation_id="mutation-1" if ok else None,
        applied_at=datetime(2026, 6, 15, tzinfo=timezone.utc) if ok else None,
        expected_subgraph_versions={"_global": 0},
        current_subgraph_versions={"_global": 0},
        new_subgraph_versions={"_global": 1} if ok else {"_global": 0},
        diff=GraphMutationDiff(edge_upserts=1, claim_keys=("claim:test",)),
        claim_keys=("claim:test",),
        verification=verification,
    )
