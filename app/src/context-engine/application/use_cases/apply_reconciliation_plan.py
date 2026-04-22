"""Deterministically apply a validated ``ReconciliationPlan``."""

from __future__ import annotations

from datetime import datetime, timezone

from domain.errors import ReconciliationApplyError
from domain.graph_mutations import ProvenanceContext, ProvenanceRef
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import MutationSummary, ReconciliationPlan, ReconciliationResult
from domain.structural_graph_mutation_applier import StructuralGraphMutationApplier

from application.use_cases.reconciliation_validation import validate_reconciliation_plan


def _build_provenance(
    plan: ReconciliationPlan,
    *,
    pot_id: str,
    episode_uuid: str | None,
    context: ProvenanceContext | None,
    graph_updated_at: datetime,
) -> ProvenanceRef:
    ctx = context or ProvenanceContext()
    return ProvenanceRef(
        pot_id=pot_id,
        source_event_id=plan.event_ref.event_id,
        episode_uuid=episode_uuid,
        source_system=plan.event_ref.source_system or None,
        source_kind=ctx.source_kind,
        source_ref=ctx.source_ref,
        event_occurred_at=ctx.event_occurred_at,
        event_received_at=ctx.event_received_at,
        graph_updated_at=graph_updated_at,
        valid_from=ctx.event_occurred_at or graph_updated_at,
        valid_to=None,
        confidence=plan.confidence,
        created_by_agent=ctx.created_by_agent,
        reconciliation_run_id=ctx.reconciliation_run_id,
    )


def apply_reconciliation_plan(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    plan: ReconciliationPlan,
    *,
    expected_pot_id: str,
    mutation_applier: GraphMutationApplierPort | None = None,
    provenance_context: ProvenanceContext | None = None,
) -> ReconciliationResult:
    """Write episodes first, then apply generic structural mutations."""
    validate_reconciliation_plan(plan, expected_pot_id)

    graph_updated_at = datetime.now(timezone.utc)

    prov = _build_provenance(
        plan,
        pot_id=expected_pot_id,
        episode_uuid=None,
        context=provenance_context,
        graph_updated_at=graph_updated_at,
    )
    episode_uuids = episodic.write_episode_drafts(expected_pot_id, plan.episodes, prov)
    primary_uuid = next((u for u in episode_uuids if u), None) or ""

    summary = MutationSummary(
        episodes_written=len([u for u in episode_uuids if u]),
    )

    try:
        applier = mutation_applier or StructuralGraphMutationApplier(structural)
        prov2 = _build_provenance(
            plan,
            pot_id=expected_pot_id,
            episode_uuid=primary_uuid or None,
            context=provenance_context,
            graph_updated_at=graph_updated_at,
        )
        summary.entity_upserts_applied = applier.apply_entity_upserts(
            expected_pot_id, plan.entity_upserts, prov2
        )
        summary.edge_upserts_applied = applier.apply_edge_upserts(
            expected_pot_id, plan.edge_upserts, prov2
        )
        summary.edge_deletes_applied = applier.apply_edge_deletes(
            expected_pot_id, plan.edge_deletes, prov2
        )
        summary.invalidations_applied = applier.apply_invalidations(
            expected_pot_id, plan.invalidations, prov2
        )
    except Exception as exc:
        raise ReconciliationApplyError(str(exc)) from exc

    return ReconciliationResult(
        ok=True,
        episode_uuids=episode_uuids,
        mutation_summary=summary,
        error=None,
        downgrades=list(plan.ontology_downgrades),
    )
