"""Apply a validated :class:`MutationBatch` to the graph.

One function, one async path. The caller produced the batch; this routine
generates a per-apply ``mutation_id``, stamps it into provenance, and
runs the four mutation verbs in order against the single
:class:`GraphWriterPort`. No episodic narrative, no sync→async bridge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from adapters.outbound.graph.writer_port import GraphWriterPort
from application.services.reconciliation_validation import (
    validate_reconciliation_plan,
)
from domain.errors import ReconciliationApplyError
from domain.graph_mutations import ProvenanceContext, ProvenanceRef
from domain.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)


def _build_provenance(
    plan: MutationBatch,
    *,
    pot_id: str,
    mutation_id: str,
    context: ProvenanceContext | None,
    graph_updated_at: datetime,
) -> ProvenanceRef:
    ctx = context or ProvenanceContext()
    # ``event_ref`` is optional now. Non-event writes can still provide a
    # deterministic source id through ProvenanceContext without fabricating an
    # EventRef; otherwise the per-apply mutation id is the fallback source.
    source_event_id = (
        plan.event_ref.event_id
        if plan.event_ref
        else ctx.source_event_id or f"mutation:{mutation_id}"
    )
    source_system = plan.event_ref.source_system if plan.event_ref else ctx.source_system
    return ProvenanceRef(
        pot_id=pot_id,
        source_event_id=source_event_id,
        mutation_id=mutation_id,
        source_system=source_system or None,
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


async def apply_mutation_batch(
    graph_writer: GraphWriterPort,
    plan: MutationBatch,
    *,
    expected_pot_id: str,
    provenance_context: ProvenanceContext | None = None,
) -> MutationResult:
    """Validate the batch, stamp provenance, and run the four mutation verbs.

    Returns a :class:`MutationResult` carrying the ``mutation_id``
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

    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=summary,
        error=None,
        downgrades=list(plan.ontology_downgrades),
    )


# Back-compat alias (Step 5a rename): callers that still import the old name
# keep working while new code uses ``apply_mutation_batch``.
apply_reconciliation_plan = apply_mutation_batch
