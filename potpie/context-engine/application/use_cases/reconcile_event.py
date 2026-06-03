"""Run reconciliation: agent plan → validate → deterministic apply → ledger updates."""

from __future__ import annotations

from domain.errors import ReconciliationApplyError, ReconciliationPlanValidationError
from domain.graph_mutations import ProvenanceContext
from domain.reconciliation_issues import validation_line_to_issue
from domain.ports.context_graph import ContextGraphPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.reconciliation import (
    MutationSummary,
    ReconciliationRequest,
    ReconciliationResult,
)
from domain.reconciliation_flags import reconciliation_enabled

from application.use_cases.agent_work_capture import (
    bind_agent_work_recorder,
    clear_agent_work_recorder,
)
from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from application.use_cases.split_reconciliation_plan import (
    split_reconciliation_plan_into_steps,
)


def _merge_mutation_summary(acc: MutationSummary, nxt: MutationSummary) -> None:
    acc.episodes_written += nxt.episodes_written
    acc.entity_upserts_applied += nxt.entity_upserts_applied
    acc.edge_upserts_applied += nxt.edge_upserts_applied
    acc.edge_deletes_applied += nxt.edge_deletes_applied
    acc.invalidations_applied += nxt.invalidations_applied
    for k, v in nxt.stamp_counts.items():
        acc.stamp_counts[k] = acc.stamp_counts.get(k, 0) + v


def reconcile_event(
    context_graph: ContextGraphPort,
    agent: ReconciliationAgentPort,
    request: ReconciliationRequest,
    *,
    reco_ledger: ReconciliationLedgerPort | None = None,
) -> ReconciliationResult:
    """Invoke agent, validate plan, apply deterministically, optionally record run metadata."""
    if not reconciliation_enabled():
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error="reconciliation_disabled",
        )

    run_id: str | None = None
    meta: dict | None = None
    if reco_ledger is not None:
        if not reco_ledger.claim_event_for_processing(request.event.event_id):
            return ReconciliationResult(
                ok=False,
                episode_uuids=[],
                mutation_summary=MutationSummary(),
                error="event_not_claimable",
            )
        attempt = reco_ledger.next_attempt_number(request.event.event_id)
        meta = agent.capability_metadata()
        run_id = reco_ledger.start_reconciliation_run(
            request.event.event_id,
            attempt_number=attempt,
            agent_name=str(meta.get("agent")) if meta else None,
            agent_version=str(meta.get("version")) if meta else None,
            toolset_version=str(meta.get("toolset_version")) if meta else None,
        )

    try:
        if reco_ledger is not None and run_id is not None:
            bind_agent_work_recorder(agent, reco_ledger, run_id)
        plan = agent.run_reconciliation(request)
        validate_reconciliation_plan(plan, request.pot_id)
        slices = split_reconciliation_plan_into_steps(plan)
        if reco_ledger is not None and run_id is not None:
            reco_ledger.record_plan_metadata(
                run_id,
                plan_summary=plan.summary,
                episode_count=len(plan.episodes),
                entity_mutation_count=len(plan.entity_upserts),
                edge_mutation_count=len(plan.edge_upserts) + len(plan.edge_deletes),
            )
        combined_summary = MutationSummary()
        episode_uuids: list[str | None] = []
        ev_actor = getattr(request.event, "actor", None)
        prov_ctx = ProvenanceContext(
            source_kind=request.event.event_type or request.event.ingestion_kind,
            source_ref=request.event.source_id or request.event.source_event_id,
            event_occurred_at=request.event.occurred_at,
            event_received_at=request.event.received_at,
            created_by_agent=str(meta.get("agent")) if meta else None,
            reconciliation_run_id=run_id,
            actor_user_id=ev_actor.user_id if ev_actor else None,
            actor_surface=ev_actor.surface if ev_actor else None,
            actor_client_name=ev_actor.client_name if ev_actor else None,
            actor_auth_method=ev_actor.auth_method if ev_actor else None,
        )
        for sl in slices:
            part = context_graph.apply_plan(
                sl,
                expected_pot_id=request.pot_id,
                provenance_context=prov_ctx,
            )
            episode_uuids.extend(part.episode_uuids)
            _merge_mutation_summary(combined_summary, part.mutation_summary)
        result = ReconciliationResult(
            ok=True,
            episode_uuids=episode_uuids,
            mutation_summary=combined_summary,
            error=None,
            downgrades=list(plan.ontology_downgrades),
        )
        if reco_ledger is not None and run_id is not None:
            reco_ledger.record_run_success(run_id)
            reco_ledger.record_event_reconciled(request.event.event_id)
        return result
    except ReconciliationPlanValidationError as exc:
        if reco_ledger is not None and run_id is not None:
            reco_ledger.record_run_failure(run_id, str(exc))
            reco_ledger.record_event_failed(request.event.event_id, str(exc))
        if exc.structured_issues:
            errs = [dict(x) for x in exc.structured_issues]
        else:
            errs = [validation_line_to_issue(str(exc))]
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error=str(exc),
            reconciliation_errors=errs,
            downgrades=[],
        )
    except ReconciliationApplyError as exc:
        if reco_ledger is not None and run_id is not None:
            reco_ledger.record_run_failure(run_id, str(exc))
            reco_ledger.record_event_failed(request.event.event_id, str(exc))
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error=str(exc),
            reconciliation_errors=[validation_line_to_issue(str(exc))],
            downgrades=[],
        )
    except Exception as exc:
        if reco_ledger is not None and run_id is not None:
            reco_ledger.record_run_failure(run_id, str(exc))
            reco_ledger.record_event_failed(request.event.event_id, str(exc))
        raise
    finally:
        clear_agent_work_recorder(agent)
