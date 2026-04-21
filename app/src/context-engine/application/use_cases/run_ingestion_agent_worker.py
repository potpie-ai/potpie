"""Background ingestion agent: plan → durable steps → enqueue per-episode apply."""

from __future__ import annotations

from typing import Any

from domain.errors import ReconciliationPlanValidationError
from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    STEP_KIND_AGENT_PLAN_SLICE,
)
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.jobs import JobEnqueuePort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation_flags import reconciliation_enabled

from application.use_cases.agent_work_capture import (
    bind_agent_work_recorder,
    clear_agent_work_recorder,
)
from application.use_cases.build_reconciliation_request import (
    build_reconciliation_request,
)
from application.use_cases.context_event_mapping import context_event_row_to_domain
from application.use_cases.reconciliation_plan_codec import reconciliation_plan_to_dict
from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from application.use_cases.split_reconciliation_plan import (
    split_reconciliation_plan_into_steps,
)


def run_ingestion_agent_for_event(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    agent: ReconciliationAgentPort,
    reco_ledger: ReconciliationLedgerPort,
    event_id: str,
    jobs: JobEnqueuePort,
    *,
    mutation_applier: GraphMutationApplierPort | None = None,
) -> dict[str, Any]:
    """
    Run planner for an agent-backed event, persist episode steps, enqueue apply jobs.

    ``mutation_applier`` is reserved for future parity with sync ``reconcile_event``; apply workers
    receive graph ports directly.
    """
    del episodic, structural, mutation_applier

    row = reco_ledger.get_event_by_id(event_id)
    if row is None:
        return {"ok": False, "error": "unknown_event"}

    existing = reco_ledger.list_episode_steps(event_id)
    if existing:
        for s in existing:
            if s.status != EPISODE_STEP_APPLIED:
                jobs.enqueue_episode_apply(row.pot_id, event_id, s.sequence)
                return {"ok": True, "status": "resumed_enqueue", "sequence": s.sequence}
        return {"ok": True, "status": "already_complete"}

    if not reconciliation_enabled():
        reco_ledger.record_event_failed(event_id, "reconciliation_disabled")
        return {"ok": False, "error": "reconciliation_disabled"}

    if not reco_ledger.claim_event_for_processing(event_id):
        return {"ok": False, "error": "not_claimable"}

    attempt = reco_ledger.next_attempt_number(event_id)
    meta = agent.capability_metadata()
    run_id = reco_ledger.start_reconciliation_run(
        event_id,
        attempt_number=attempt,
        agent_name=str(meta.get("agent")) if meta else None,
        agent_version=str(meta.get("version")) if meta else None,
        toolset_version=str(meta.get("toolset_version")) if meta else None,
    )

    try:
        bind_agent_work_recorder(agent, reco_ledger, run_id)
        event = context_event_row_to_domain(row)
        request = build_reconciliation_request(event)
        plan = agent.run_reconciliation(request)
        validate_reconciliation_plan(plan, request.pot_id)
        slices = split_reconciliation_plan_into_steps(plan)
        reco_ledger.record_plan_metadata(
            run_id,
            plan_summary=plan.summary,
            episode_count=len(plan.episodes),
            entity_mutation_count=len(plan.entity_upserts),
            edge_mutation_count=len(plan.edge_upserts) + len(plan.edge_deletes),
        )
        reco_ledger.record_run_plan_json(run_id, reconciliation_plan_to_dict(plan))
        steps_payload: list[tuple[int, str, dict[str, Any]]] = []
        for i, sl in enumerate(slices, start=1):
            steps_payload.append(
                (i, STEP_KIND_AGENT_PLAN_SLICE, reconciliation_plan_to_dict(sl))
            )
        reco_ledger.replace_episode_steps_for_event(
            row.pot_id, event_id, run_id, steps_payload
        )
        reco_ledger.record_run_success(run_id)
        reco_ledger.mark_event_episodes_queued(event_id)
        for i, _, _ in steps_payload:
            jobs.enqueue_episode_apply(row.pot_id, event_id, i)
        return {"ok": True, "steps_enqueued": len(steps_payload), "event_id": event_id}
    except (ReconciliationPlanValidationError, ValueError) as exc:
        reco_ledger.record_run_failure(run_id, str(exc))
        reco_ledger.record_event_failed(event_id, str(exc))
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        reco_ledger.record_run_failure(run_id, str(exc))
        reco_ledger.record_event_failed(event_id, str(exc))
        raise
    finally:
        clear_agent_work_recorder(agent)
