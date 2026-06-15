"""Job-runner adapter: thin shim between the Celery task and the verb.

One background job exists: a per-batch processor (``handle_process_batch``).
It is session-scoped and rebuilds the container per call so host-side
session-bound resolvers (e.g. ``SqlalchemyPotResolution``) stay valid.

Backfill is no longer a standalone enumerate-then-submit sweep. A source
attach (GitHub ``repository.added`` / Linear ``linear_team.added``) emits a
single ``agent_reconciliation`` event; the reconciliation agent — planner
on, via the backfill playbooks — enumerates and seeds the graph through this
same per-batch path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from application.use_cases.process_batch import process_batch
from bootstrap import sentry_metrics_runtime
from bootstrap.ingestion_server import IngestionServerContainer
from bootstrap.observability_context import correlation_scope
from bootstrap.observability_runtime import get_observability
from domain.ports.observability import SPAN_KIND_CONSUMER


def _ingress_links(reco_ledger, batches_repo, batch_id: str) -> list[str]:
    """W3C traceparents of the batch's events' (long-gone) ingress traces.

    The batch span links back to each so an operator can pivot from the
    async run to the request that produced an event, across the windowed
    delay that makes live context propagation impossible.
    """
    links: list[str] = []
    try:
        for ref in batches_repo.list_events_for_batch(batch_id):
            row = reco_ledger.get_event_by_id(ref.event_id)
            tp = getattr(row, "correlation_id", None)
            if tp:
                links.append(tp)
    except Exception:  # noqa: BLE001 — links are best-effort
        return links
    return links


def handle_process_batch(
    db: Session,
    batch_id: str,
    *,
    build_ingestion_server: Callable[[Session], IngestionServerContainer],
) -> dict[str, Any]:
    """Claim one batch by id and run the reconciliation agent over it.

    Called by the host worker once per ``jobs.enqueue_batch`` event. If the
    batch is already claimed/running/done (a redundant enqueue races and
    loses), ``claim_batch_by_id`` returns ``None`` and this is a no-op.
    """
    container = build_ingestion_server(db)
    if container.reconciliation_agent is None:
        sentry_metrics_runtime.count(
            "ce.batch.finished_total",
            1,
            attributes={"result": "skipped_no_reconciliation_agent"},
        )
        return {"status": "skipped", "reason": "no_reconciliation_agent"}
    batches_repo = container.batch_repository(db)
    batch = batches_repo.claim_batch_by_id(batch_id)
    if batch is None:
        sentry_metrics_runtime.count(
            "ce.batch.finished_total",
            1,
            attributes={"result": "skipped_not_pending"},
        )
        return {"status": "skipped", "reason": "not_pending", "batch_id": batch_id}

    reco_ledger = container.reconciliation_ledger(db)
    obs = get_observability()
    links = _ingress_links(reco_ledger, batches_repo, batch.id)
    # The batch is the primary async trace (fan-in: N events → 1 run → M
    # mutations). It links back to each event's ingress trace.
    with correlation_scope(batch_id=batch.id, pot_id=batch.pot_id):
        with obs.span(
            "batch.process",
            kind=SPAN_KIND_CONSUMER,
            attributes={"batch_id": batch.id, "pot_id": batch.pot_id},
            links=links,
        ) as span:
            obs.counter(
                "ce.batch.started_total", 1, attributes={"pot_id": batch.pot_id}
            )
            sentry_metrics_runtime.count(
                "ce.batch.started_total",
                1,
                attributes={"result": "started"},
            )
            # Time-in-pending: the windowed-5min canary. If the flusher
            # wedges, this is what screams before anything else.
            try:
                if batch.created_at and batch.claimed_at:
                    wait_ms = (
                        batch.claimed_at - batch.created_at
                    ).total_seconds() * 1000.0
                    obs.histogram(
                        "ce.batch.time_in_pending_ms",
                        wait_ms,
                        attributes={"pot_id": batch.pot_id},
                    )
                    sentry_metrics_runtime.distribution(
                        "ce.batch.time_in_pending_ms",
                        wait_ms,
                        unit="millisecond",
                        attributes={"result": "started"},
                    )
            except Exception:  # noqa: BLE001 — best-effort metric
                pass
            outcome = process_batch(
                batch=batch,
                agent=container.reconciliation_agent,
                batches=batches_repo,
                reco_ledger=reco_ledger,
                checkpoints=container.agent_checkpoint_store(db),
                pots=container.pots,
                policy=container.policy(),
                stream_publisher=container.event_stream_publisher,
                execution_log=container.agent_execution_log(db),
            )
            span.set_attribute("batch.ok", bool(outcome.ok))
            span.set_attribute(
                "batch.completed_events", len(outcome.completed_event_ids)
            )
            span.set_attribute("batch.tool_calls", outcome.tool_call_count)
            if not outcome.ok:
                span.set_error(outcome.error or "batch failed")
            obs.counter(
                "ce.batch.finished_total",
                1,
                attributes={
                    "pot_id": batch.pot_id,
                    "result": "ok" if outcome.ok else "failed",
                },
            )
            sentry_metrics_runtime.count(
                "ce.batch.finished_total",
                1,
                attributes={"result": "ok" if outcome.ok else "failed"},
            )
    return {
        "status": "ok" if outcome.ok else "failed",
        "batch_id": outcome.batch_id,
        "completed_event_ids": outcome.completed_event_ids,
        "tool_call_count": outcome.tool_call_count,
        "error": outcome.error,
    }
