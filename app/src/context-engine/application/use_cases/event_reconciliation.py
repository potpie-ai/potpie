"""Record canonical events and run reconciliation (HTTP / workers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from domain.context_events import ContextEvent, EventScope
from domain.ports.context_graph import ContextGraphPort
from domain.ports.jobs import JobEnqueuePort, NoOpJobEnqueue
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ingestion_event_models import IngestionEvent
from domain.ports.reconciliation_ledger import ContextEventRow, ReconciliationLedgerPort
from domain.reconciliation import ReconciliationResult

from application.use_cases.build_reconciliation_request import build_reconciliation_request
from application.use_cases.record_context_event import record_context_event
from application.use_cases.reconcile_event import reconcile_event


@dataclass(slots=True)
class RecordReconcileOutcome:
    """Result of persisting an event and optionally running reconciliation."""

    inserted: bool
    event_id: str
    reconciliation: ReconciliationResult | None
    job_id: str | None = None


def record_and_reconcile_context_event(
    context_graph: ContextGraphPort,
    agent: ReconciliationAgentPort,
    reco_ledger: ReconciliationLedgerPort,
    scope: EventScope,
    event: ContextEvent,
    *,
    sync: bool = True,
    jobs: JobEnqueuePort | None = None,
) -> RecordReconcileOutcome:
    """Append event; sync path runs ``reconcile_event``; async path enqueues ingestion work."""
    event_id, inserted = record_context_event(reco_ledger, scope, event)
    if not inserted:
        return RecordReconcileOutcome(
            inserted=False,
            event_id=event_id,
            reconciliation=None,
        )
    jq = jobs or NoOpJobEnqueue()
    if not sync:
        reco_ledger.mark_event_queued(event_id)
        jid = str(uuid4())
        reco_ledger.set_event_job_metadata(event_id, job_id=jid, correlation_id=jid)
        jq.enqueue_ingestion_event(
            event_id,
            pot_id=event.pot_id,
            kind=event.ingestion_kind or "agent_reconciliation",
        )
        if isinstance(jq, NoOpJobEnqueue):
            from application.use_cases.run_ingestion_agent_worker import run_ingestion_agent_for_event

            run_ingestion_agent_for_event(
                agent,
                reco_ledger,
                event_id,
                jq,
            )
        return RecordReconcileOutcome(
            inserted=True,
            event_id=event_id,
            reconciliation=None,
            job_id=jid,
        )
    request = build_reconciliation_request(event)
    result = reconcile_event(
        context_graph,
        agent,
        request,
        reco_ledger=reco_ledger,
    )
    return RecordReconcileOutcome(
        inserted=True,
        event_id=event_id,
        reconciliation=result,
    )


def context_event_row_to_payload(
    row: ContextEventRow,
    *,
    episode_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """JSON-serializable view of a stored context event."""
    out: dict[str, Any] = {
        "id": row.id,
        "pot_id": row.pot_id,
        "provider": row.provider,
        "provider_host": row.provider_host,
        "repo_name": row.repo_name,
        "source_system": row.source_system,
        "event_type": row.event_type,
        "action": row.action,
        "source_id": row.source_id,
        "source_event_id": row.source_event_id,
        "payload": row.payload,
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
        "received_at": row.received_at.isoformat(),
        "status": row.status,
        "ingestion_kind": row.ingestion_kind,
        "job_id": row.job_id,
        "correlation_id": row.correlation_id,
        "idempotency_key": row.idempotency_key,
        "source_channel": row.source_channel,
        "actor": row.actor.to_payload() if row.actor else None,
    }
    if episode_steps is not None:
        out["episode_steps"] = episode_steps
    return out


def ingestion_event_to_payload(
    ev: IngestionEvent,
    *,
    episode_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """JSON view aligned with :func:`context_event_row_to_payload` (read model / EventQueryService)."""
    legacy = ev.raw_status or ev.status
    out: dict[str, Any] = {
        "id": ev.event_id,
        "pot_id": ev.pot_id,
        "provider": ev.provider,
        "provider_host": ev.provider_host,
        "repo_name": ev.repo_name,
        "source_system": ev.source_system,
        "event_type": ev.event_type,
        "action": ev.action,
        "source_id": ev.source_id,
        "source_event_id": ev.source_event_id,
        "payload": ev.payload,
        "occurred_at": ev.occurred_at.isoformat() if ev.occurred_at else None,
        "received_at": ev.submitted_at.isoformat(),
        "status": legacy,
        "lifecycle_status": ev.status,
        "ingestion_kind": ev.ingestion_kind,
        "source_channel": ev.source_channel,
        "dedup_key": ev.dedup_key,
        "stage": ev.stage,
        "step_total": ev.step_total,
        "step_done": ev.step_done,
        "step_error": ev.step_error,
        "error": ev.error,
        "job_id": ev.job_id,
        "correlation_id": ev.correlation_id,
        "idempotency_key": ev.idempotency_key,
        "metadata": ev.metadata,
        "actor": ev.actor.to_payload() if ev.actor else None,
    }
    if episode_steps is not None:
        out["episode_steps"] = episode_steps
    return out
