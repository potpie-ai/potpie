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

from potpie_context_core.context_events import ContextEvent, EventScope
from potpie_context_engine.domain.ports.batch_repository import BatchRepositoryPort
from potpie_context_engine.domain.ports.context_graph_job_queue import ContextGraphJobQueuePort
from potpie_context_engine.domain.ports.ingestion_config import IngestionConfigPort
from potpie_context_engine.domain.ports.reconciliation_ledger import ReconciliationLedgerPort

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EventAdmissionOutcome:
    event_id: str
    batch_id: str | None
    inserted: bool
    """``False`` when the event was a duplicate (idempotent re-submission)."""
    enqueued: bool = True
    """``False`` when the pot is in ``windowed`` mode — the batch is kept
    pending and a periodic flusher will enqueue it later."""


def admit_event(
    reco_ledger: ReconciliationLedgerPort,
    batches: BatchRepositoryPort,
    jobs: ContextGraphJobQueuePort,
    scope: EventScope,
    event: ContextEvent,
    *,
    ingestion_config: IngestionConfigPort | None = None,
) -> EventAdmissionOutcome:
    """Append the event, coalesce it into the pot's pending batch, and (maybe) enqueue.

    Duplicate events (same scope + ``source_id``) short-circuit with
    ``inserted=False`` — a producer retrying the same payload does not
    retrigger work.

    Enqueue is conditional on the pot's ingestion mode:

    - ``immediate`` (legacy): enqueue right away. If the worker is already
      processing the coalesced batch, ``claim_batch_by_id`` returns ``None``
      on the duplicate job and the redundant call no-ops.
    - ``windowed`` (default since Phase 4): keep the batch pending. A
      scheduled task closes-and-enqueues batches older than the configured
      window. The user can force-flush via ``/ingest/flush``.

    Enqueue failures are best-effort: the batch is durable in Postgres, so
    the windowed flusher (or the next event) will pick it up.
    """
    event_id, inserted = reco_ledger.append_event(scope, event)
    if not inserted:
        return EventAdmissionOutcome(event_id=event_id, batch_id=None, inserted=False)
    reco_ledger.mark_event_queued(event_id)
    batch_id = batches.upsert_open_batch_for_pot(event.pot_id, event_id)

    # Default to immediate when no config port is wired — preserves legacy
    # behaviour for callers that haven't been updated.
    mode = "immediate"
    if ingestion_config is not None:
        try:
            mode = ingestion_config.get(event.pot_id).mode
        except Exception:
            logger.warning(
                "admit_event: failed to read ingestion config for pot %s; "
                "defaulting to immediate",
                event.pot_id,
                exc_info=True,
            )

    if mode == "windowed":
        # Batch stays pending; periodic flush task picks it up.
        return EventAdmissionOutcome(
            event_id=event_id,
            batch_id=batch_id,
            inserted=True,
            enqueued=False,
        )

    try:
        jobs.enqueue_batch(batch_id)
    except Exception:
        logger.exception(
            "admit_event: enqueue_batch failed for batch %s (event %s); "
            "batch is durable, next event will re-enqueue",
            batch_id,
            event_id,
        )
    return EventAdmissionOutcome(
        event_id=event_id,
        batch_id=batch_id,
        inserted=True,
        enqueued=True,
    )
