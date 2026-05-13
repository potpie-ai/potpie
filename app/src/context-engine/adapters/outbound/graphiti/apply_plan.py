"""Deterministically apply a validated ``ReconciliationPlan``.

Phase 1 of the rebuild made Graphiti the sole writer. Episodic drafts
land via :meth:`EpisodicGraphPort.write_episode_drafts`, and canonical
entity/edge mutations land via the same port's ``apply_*`` methods,
which run through Graphiti's driver. There is exactly one downstream
write surface here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from adapters.outbound.graphiti.port import EpisodicGraphPort
from application.services.reconciliation_validation import validate_reconciliation_plan
from domain.errors import ReconciliationApplyError
from domain.graph_mutations import ProvenanceContext, ProvenanceRef
from domain.reconciliation import MutationSummary, ReconciliationPlan, ReconciliationResult


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
        actor_user_id=ctx.actor_user_id,
        actor_surface=ctx.actor_surface,
        actor_client_name=ctx.actor_client_name,
        actor_auth_method=ctx.actor_auth_method,
    )


def apply_reconciliation_plan(
    episodic: EpisodicGraphPort,
    plan: ReconciliationPlan,
    *,
    expected_pot_id: str,
    provenance_context: ProvenanceContext | None = None,
) -> ReconciliationResult:
    """Write episodes first, then apply canonical mutations through Graphiti's driver."""
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
        prov2 = _build_provenance(
            plan,
            pot_id=expected_pot_id,
            episode_uuid=primary_uuid or None,
            context=provenance_context,
            graph_updated_at=graph_updated_at,
        )
        summary.entity_upserts_applied = episodic.apply_entity_upserts(
            expected_pot_id, plan.entity_upserts, prov2
        )
        summary.edge_upserts_applied = episodic.apply_edge_upserts(
            expected_pot_id, plan.edge_upserts, prov2
        )
        summary.edge_deletes_applied = episodic.apply_edge_deletes(
            expected_pot_id, plan.edge_deletes, prov2
        )
        summary.invalidations_applied = episodic.apply_invalidations(
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


async def apply_reconciliation_plan_async(
    episodic: EpisodicGraphPort,
    plan: ReconciliationPlan,
    *,
    expected_pot_id: str,
    provenance_context: ProvenanceContext | None = None,
) -> ReconciliationResult:
    """Async-native plan apply.

    Use this from inside an event loop (agent tools, FastAPI handlers).
    It avoids the sync→async→sync bridge in ``GraphitiEpisodicAdapter._sync_run``
    that was crashing with ``Future attached to a different loop`` when called
    from a worker thread whose Graphiti client was bound to an earlier loop.
    """
    validate_reconciliation_plan(plan, expected_pot_id)

    graph_updated_at = datetime.now(timezone.utc)

    prov = _build_provenance(
        plan,
        pot_id=expected_pot_id,
        episode_uuid=None,
        context=provenance_context,
        graph_updated_at=graph_updated_at,
    )
    episode_uuids = await episodic.write_episode_drafts_async(
        expected_pot_id, plan.episodes, prov
    )
    primary_uuid = next((u for u in episode_uuids if u), None) or ""

    summary = MutationSummary(
        episodes_written=len([u for u in episode_uuids if u]),
    )

    try:
        prov2 = _build_provenance(
            plan,
            pot_id=expected_pot_id,
            episode_uuid=primary_uuid or None,
            context=provenance_context,
            graph_updated_at=graph_updated_at,
        )
        summary.entity_upserts_applied = await episodic.apply_entity_upserts_async(
            expected_pot_id, plan.entity_upserts, prov2
        )
        summary.edge_upserts_applied = await episodic.apply_edge_upserts_async(
            expected_pot_id, plan.edge_upserts, prov2
        )
        summary.edge_deletes_applied = await episodic.apply_edge_deletes_async(
            expected_pot_id, plan.edge_deletes, prov2
        )
        summary.invalidations_applied = await episodic.apply_invalidations_async(
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
