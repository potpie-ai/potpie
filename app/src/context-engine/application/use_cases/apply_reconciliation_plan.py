"""Deterministically apply a validated ``ReconciliationPlan``."""

from __future__ import annotations

from domain.errors import ReconciliationApplyError
from domain.graph_mutations import ProvenanceRef
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import MutationSummary, ReconciliationPlan, ReconciliationResult
from domain.structural_graph_mutation_applier import StructuralGraphMutationApplier

from application.use_cases.reconciliation_validation import validate_reconciliation_plan


def apply_reconciliation_plan(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    plan: ReconciliationPlan,
    *,
    expected_pot_id: str,
    mutation_applier: GraphMutationApplierPort | None = None,
) -> ReconciliationResult:
    """Write episodes first, then structural mutations (compat or generic)."""
    validate_reconciliation_plan(plan, expected_pot_id)

    prov = ProvenanceRef(
        pot_id=expected_pot_id,
        source_event_id=plan.event_ref.event_id,
        episode_uuid=None,
    )
    episode_uuids = episodic.write_episode_drafts(expected_pot_id, plan.episodes, prov)
    primary_uuid = next((u for u in episode_uuids if u), None) or ""

    summary = MutationSummary(
        episodes_written=len([u for u in episode_uuids if u]),
    )

    try:
        if plan.compat_github_pr_merged is not None:
            c = plan.compat_github_pr_merged
            raw_pn = c.pr_data.get("number")
            if raw_pn is None:
                raise ReconciliationApplyError("compat plan requires pr_data.number")
            pr_number = int(raw_pn)
            stamp_counts = structural.stamp_pr_entities(
                pot_id=expected_pot_id,
                episode_uuid=primary_uuid,
                repo_name=c.repo_name,
                pr_number=pr_number,
                commits=c.commits,
                review_threads=c.review_threads,
                pr_data=c.pr_data,
                author=c.pr_data.get("author"),
                pr_title=c.pr_data.get("title"),
                issue_comments=c.issue_comments or [],
            )
            summary.stamp_counts = stamp_counts
        else:
            applier = mutation_applier or StructuralGraphMutationApplier(structural)
            prov2 = ProvenanceRef(
                pot_id=expected_pot_id,
                source_event_id=plan.event_ref.event_id,
                episode_uuid=primary_uuid or None,
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
