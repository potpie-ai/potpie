"""Split a full ``ReconciliationPlan`` into ordered per-episode apply slices."""

from __future__ import annotations

from domain.reconciliation import ReconciliationPlan


def split_reconciliation_plan_into_steps(plan: ReconciliationPlan) -> list[ReconciliationPlan]:
    """
    One slice per episode; structural mutations and compat path attach only to the last slice.

    This matches the ingestion async contract: queue and apply episodes in sequence without
    skipping structural work that belongs to the full reconciliation.
    """
    n = len(plan.episodes)
    if n == 0:
        return []

    out: list[ReconciliationPlan] = []
    for i in range(n):
        is_last = i == n - 1
        out.append(
            ReconciliationPlan(
                event_ref=plan.event_ref,
                summary=f"{plan.summary} (step {i + 1}/{n})",
                episodes=[plan.episodes[i]],
                entity_upserts=list(plan.entity_upserts) if is_last else [],
                edge_upserts=list(plan.edge_upserts) if is_last else [],
                edge_deletes=list(plan.edge_deletes) if is_last else [],
                invalidations=list(plan.invalidations) if is_last else [],
                evidence=list(plan.evidence) if is_last else [],
                confidence=plan.confidence if is_last else None,
                warnings=list(plan.warnings) if is_last else [],
                compat_github_pr_merged=plan.compat_github_pr_merged if is_last else None,
                ontology_downgrades=[],
            )
        )
    return out
