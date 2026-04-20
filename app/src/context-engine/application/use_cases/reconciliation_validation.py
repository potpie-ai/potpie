"""Validate ``ReconciliationPlan`` before deterministic apply."""

from __future__ import annotations

from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import InvalidationOp
from domain.ontology import is_canonical_edge_type, validate_structural_mutations
from domain.reconciliation import ReconciliationPlan

MAX_EPISODES = 32
MAX_GENERIC_ENTITY_UPSERTS = 5000
MAX_GENERIC_EDGES = 10000
MAX_INVALIDATIONS = 2000


def validate_reconciliation_plan(
    plan: ReconciliationPlan, expected_pot_id: str
) -> None:
    if plan.event_ref.pot_id != expected_pot_id:
        raise ReconciliationPlanValidationError(
            "plan event_ref.pot_id does not match expected pot"
        )

    if len(plan.episodes) > MAX_EPISODES:
        raise ReconciliationPlanValidationError("too many episodes in plan")

    if plan.compat_github_pr_merged is not None:
        if (
            plan.entity_upserts
            or plan.edge_upserts
            or plan.edge_deletes
            or plan.invalidations
        ):
            raise ReconciliationPlanValidationError(
                "compat_github_pr_merged cannot be combined with generic structural mutations"
            )
        return

    if len(plan.entity_upserts) > MAX_GENERIC_ENTITY_UPSERTS:
        raise ReconciliationPlanValidationError("entity upsert cap exceeded")
    if len(plan.edge_upserts) + len(plan.edge_deletes) > MAX_GENERIC_EDGES:
        raise ReconciliationPlanValidationError("edge mutation cap exceeded")
    if len(plan.invalidations) > MAX_INVALIDATIONS:
        raise ReconciliationPlanValidationError("invalidation cap exceeded")

    ontology_errors = validate_structural_mutations(
        plan.entity_upserts,
        plan.edge_upserts,
        plan.edge_deletes,
    )
    ontology_errors.extend(_validate_invalidations(plan.invalidations))
    if ontology_errors:
        sample = "; ".join(ontology_errors[:8])
        suffix = (
            ""
            if len(ontology_errors) <= 8
            else f"; ... {len(ontology_errors) - 8} more"
        )
        raise ReconciliationPlanValidationError(
            f"ontology validation failed: {sample}{suffix}"
        )


def _validate_invalidations(items: list[InvalidationOp]) -> list[str]:
    errors: list[str] = []
    for item in items:
        if not item.target_entity_key and not item.target_edge:
            errors.append("invalidation must set target_entity_key or target_edge")
        if item.target_edge:
            edge_type = item.target_edge[0]
            if not is_canonical_edge_type(edge_type):
                errors.append(
                    f"invalidation target_edge has non-canonical edge_type: {edge_type!r}"
                )
    return errors
