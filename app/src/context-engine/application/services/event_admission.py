"""Ledger primitive: admit a ``ContextEvent`` and enqueue its batch.

Used internally by :class:`DefaultIngestionSubmissionService`. Inbound
adapters do not call this directly — they go through the service's
``submit(IngestionSubmissionRequest)`` method, which is the agent-facing
canonical inbound after Phase 4.

The event is recorded in ``context_events`` (idempotent on scope +
source_id), coalesced into the pot's open ``pending`` batch (a new batch
is created if none is pending), and the batch is enqueued for the
reconciliation worker via ``jobs.enqueue_batch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.context_events import ContextEvent, EventScope
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.context_graph_job_queue import ContextGraphJobQueuePort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EventAdmissionOutcome:
    event_id: str
    batch_id: str | None
    inserted: bool
    """``False`` when the event was a duplicate (idempotent re-submission)."""


def admit_event(
    reco_ledger: ReconciliationLedgerPort,
    batches: BatchRepositoryPort,
    jobs: ContextGraphJobQueuePort,
    scope: EventScope,
    event: ContextEvent,
) -> EventAdmissionOutcome:
    """Append the event, coalesce it into the pot's pending batch, and enqueue.

    Duplicate events (same scope + ``source_id``) short-circuit with
    ``inserted=False`` and no batch update / enqueue — a producer retrying
    the same payload should not retrigger work.

    The enqueue is fire-and-forget. If the worker is already processing the
    coalesced batch when this fires, ``claim_batch_by_id`` will return
    ``None`` on the duplicate job and the redundant call no-ops. If the
    enqueue itself fails (broker outage), we log and continue — the event
    and batch are already durable in Postgres, and a follow-up event will
    trigger another enqueue that picks up both.
    """
    event_id, inserted = reco_ledger.append_event(scope, event)
    if not inserted:
        return EventAdmissionOutcome(event_id=event_id, batch_id=None, inserted=False)
    reco_ledger.mark_event_queued(event_id)
    batch_id = batches.upsert_open_batch_for_pot(event.pot_id, event_id)
    try:
        jobs.enqueue_batch(batch_id)
    except Exception:
        logger.exception(
            "admit_event: enqueue_batch failed for batch %s (event %s); "
            "batch is durable, next event will re-enqueue",
            batch_id,
            event_id,
        )
    return EventAdmissionOutcome(event_id=event_id, batch_id=batch_id, inserted=True)
