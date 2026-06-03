"""Apply a validated :class:`ReconciliationPlan` to a Cypher-speaking backend.

One function, one async path. The agent produced the plan; this routine
generates a per-apply ``mutation_id``, stamps it into provenance, and
runs the four mutation verbs in order against a single
:class:`GraphWriterPort`. The choreography is identical for every Cypher
backend (Neo4j, FalkorDB) — they differ in `enabled`, `ensure_indexes`,
and `reset_pot` (which this function never calls), so the apply path
genuinely is shared. Non-Cypher backends (in-memory) implement
``GraphMutationPort.apply`` directly and do not call this.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from adapters.outbound.graph.backends._cypher_shared.writer import GraphWriterPort
from application.services.reconciliation_validation import (
    validate_reconciliation_plan,
)
from domain.errors import ReconciliationApplyError
from domain.graph_mutations import ProvenanceContext, ProvenanceRef
from domain.reconciliation import (
    MutationSummary,
    ReconciliationPlan,
    ReconciliationResult,
)


def _build_provenance(
    plan: ReconciliationPlan,
    *,
    pot_id: str,
    mutation_id: str,
    context: ProvenanceContext | None,
    graph_updated_at: datetime,
) -> ProvenanceRef:
    ctx = context or ProvenanceContext()
    return ProvenanceRef(
        pot_id=pot_id,
        source_event_id=plan.event_ref.event_id,
        mutation_id=mutation_id,
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


async def apply_reconciliation_plan(
    graph_writer: GraphWriterPort,
    plan: ReconciliationPlan,
    *,
    expected_pot_id: str,
    provenance_context: ProvenanceContext | None = None,
) -> ReconciliationResult:
    """Validate the plan, stamp provenance, and run the four mutation verbs.

    Returns a :class:`ReconciliationResult` carrying the ``mutation_id``
    (a per-apply UUID) so callers can later trace every produced edge
    back to this apply.
    """
    validate_reconciliation_plan(plan, expected_pot_id)

    mutation_id = str(uuid4())
    graph_updated_at = datetime.now(timezone.utc)
    prov = _build_provenance(
        plan,
        pot_id=expected_pot_id,
        mutation_id=mutation_id,
        context=provenance_context,
        graph_updated_at=graph_updated_at,
    )

    summary = MutationSummary()
    try:
        summary.entity_upserts_applied = await graph_writer.upsert_entities(
            expected_pot_id, plan.entity_upserts, prov
        )
        summary.edge_upserts_applied = await graph_writer.upsert_edges(
            expected_pot_id, plan.edge_upserts, prov
        )
        summary.edge_deletes_applied = await graph_writer.delete_edges(
            expected_pot_id, plan.edge_deletes, prov
        )
        summary.invalidations_applied = await graph_writer.invalidate(
            expected_pot_id, plan.invalidations, prov
        )
    except Exception as exc:
        raise ReconciliationApplyError(str(exc)) from exc

    return ReconciliationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=summary,
        error=None,
        downgrades=list(plan.ontology_downgrades),
    )


__all__ = ["apply_reconciliation_plan"]
