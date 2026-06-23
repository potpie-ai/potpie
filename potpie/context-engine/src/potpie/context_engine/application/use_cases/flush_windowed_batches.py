"""Periodically close-and-enqueue windowed batches.

In ``windowed`` ingestion mode events accumulate in the pot's open batch
without enqueueing. This task picks up every pot whose open batch is older
than the configured ``window_minutes`` (or above ``min_batch_size``) and
enqueues it. Closing the batch is implicit — once enqueued, the worker
claims it and the next event admission opens a fresh ``pending`` batch.

The task is idempotent: re-running it within a single window only enqueues
batches whose timer has elapsed since the last successful enqueue. Workers
that race on the same batch lose at ``claim_batch_by_id`` (returns ``None``)
so duplicate enqueues are safe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from potpie.context_engine.domain.ports.batch_repository import BatchRepositoryPort
from potpie.context_engine.domain.ports.context_graph_job_queue import ContextGraphJobQueuePort
from potpie.context_engine.domain.ports.ingestion_config import IngestionConfigPort

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FlushOutcome:
    pot_ids_flushed: list[str]
    batches_enqueued: int
    errors: int


def flush_ready_windowed_pots(
    *,
    config: IngestionConfigPort,
    batches: BatchRepositoryPort,
    jobs: ContextGraphJobQueuePort,
    now_unix_seconds: float | None = None,
) -> FlushOutcome:
    """Enqueue every windowed pot whose open batch is ready to flush.

    ``now_unix_seconds`` is injected for testability. In production it
    defaults to ``time.time()`` so the SQL ``EXTRACT(EPOCH FROM …)``
    comparison anchors to wall-clock.
    """
    as_of = now_unix_seconds if now_unix_seconds is not None else time.time()
    ready = config.list_windowed_pots_ready_to_flush(as_of_unix_seconds=as_of)

    flushed_pot_ids: list[str] = []
    errors = 0
    for cfg in ready:
        try:
            batch_id = batches.get_open_batch_id_for_pot(cfg.pot_id)
        except Exception:
            logger.exception(
                "flush_windowed: failed to read open batch for pot %s", cfg.pot_id
            )
            errors += 1
            continue
        if batch_id is None:
            # Race: another flusher already enqueued or the batch closed.
            continue
        try:
            jobs.enqueue_batch(batch_id)
            flushed_pot_ids.append(cfg.pot_id)
        except Exception:
            logger.exception(
                "flush_windowed: enqueue_batch failed for batch %s (pot %s)",
                batch_id,
                cfg.pot_id,
            )
            errors += 1

    return FlushOutcome(
        pot_ids_flushed=flushed_pot_ids,
        batches_enqueued=len(flushed_pot_ids),
        errors=errors,
    )


def force_flush_pot(
    *,
    pot_id: str,
    batches: BatchRepositoryPort,
    jobs: ContextGraphJobQueuePort,
) -> str | None:
    """Enqueue the pot's open batch right now (the user-facing 'force ingest').

    Returns the enqueued batch_id, or ``None`` when the pot has no pending
    batch. Works for both ingestion modes — useful as an escape hatch in
    immediate mode too if a previous enqueue dropped on the floor.
    """
    batch_id = batches.get_open_batch_id_for_pot(pot_id)
    if batch_id is None:
        return None
    try:
        jobs.enqueue_batch(batch_id)
    except Exception:
        logger.exception(
            "force_flush_pot: enqueue_batch failed for batch %s (pot %s)",
            batch_id,
            pot_id,
        )
        # Caller decides whether to surface as 5xx — we still return the
        # batch_id so the user sees something durable was created.
    return batch_id
