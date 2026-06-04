"""Celery tasks for context-graph ingestion workflows."""

import os

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from application.use_cases.context_graph_jobs import handle_process_batch
from application.use_cases.flush_windowed_batches import (
    flush_ready_windowed_pots,
)
from application.use_cases.reap_stale_batches import reap_stale_batches


def _stale_batch_lease_seconds() -> float:
    """Lease after which an in-flight batch is presumed dead.

    Must exceed Celery's hard ``task_time_limit`` so the reaper can never
    race a still-alive (merely slow) run — Celery kills a live task at the
    time limit, so anything older than this lease is definitively dead.
    Defaults to ``CELERY_TASK_TIME_LIMIT`` + 15 min of headroom.
    """
    task_time_limit = int(os.getenv("CELERY_TASK_TIME_LIMIT", "5400"))
    return float(
        os.getenv("CELERY_STALE_BATCH_LEASE_SECS", str(task_time_limit + 900))
    )


class ContextGraphTask(Task):
    """Base task class with managed DB session."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


# Backfill is no longer a standalone Celery task. GitHub repo attach
# (``repository.added``) and Linear team attach (``linear_team.added``)
# each emit one ``agent_reconciliation`` event; the reconciliation agent
# (planner on, backfill playbooks) enumerates and seeds the graph via
# ``context_graph_process_batch`` — the same path live webhooks take.


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_process_batch",
    queue="context-graph-etl",
    # Bounded autoretry for transient infra faults (DB blip, etc.) that
    # escape ``handle_process_batch`` *before* the batch is claimed —
    # ``process_batch`` already converts agent failures into a terminal
    # ``failed`` internally, and a retry that arrives after the batch left
    # ``pending`` is a safe no-op (``claim_batch_by_id`` returns ``None``).
    # Bounded (not infinite) on purpose — the stale-batch reaper, not
    # endless redelivery, is the backstop for a worker that dies mid-run.
    autoretry_for=(Exception,),
    max_retries=int(os.getenv("CONTEXT_GRAPH_PROCESS_BATCH_MAX_RETRIES", "3")),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def context_graph_process_batch(self, batch_id: str) -> dict:
    """Event-triggered: claim one batch by id and run the reconciliation agent.

    Enqueued by ``CeleryContextGraphJobQueue.enqueue_batch`` from inside
    ``admit_event``. Redundant enqueues for an already-claimed batch are
    no-ops on the worker side.
    """
    return handle_process_batch(
        self.db,
        batch_id,
        build_container=build_container_for_session,
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_flush_windowed_batches",
    queue="context-graph-etl",
)
def context_graph_flush_windowed_batches(self) -> dict:
    """Beat-scheduled: enqueue any windowed pots whose open batch is ripe.

    Runs every minute (see ``celery_app.conf.beat_schedule``). Idempotent:
    re-running within a window only re-enqueues already-claimed batches,
    which the worker rejects via ``claim_batch_by_id``.
    """
    container = build_container_for_session(self.db)
    outcome = flush_ready_windowed_pots(
        config=container.ingestion_config(self.db),
        batches=container.batch_repository(self.db),
        jobs=container.jobs,
    )
    return {
        "pot_ids_flushed": outcome.pot_ids_flushed,
        "batches_enqueued": outcome.batches_enqueued,
        "errors": outcome.errors,
    }


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_reap_stale_batches",
    queue="context-graph-etl",
)
def context_graph_reap_stale_batches(self) -> dict:
    """Beat-scheduled: fail batches stuck in-flight past the lease.

    Recovers the one failure the rest of the pipeline cannot: a worker that
    died mid-run (OOM, pod restart, hard time-limit). Such a batch is never
    redelivered and never re-claimable, so without this its events sit at
    ``processing`` forever. Idempotent — ``mark_batch_failed`` moves the row
    out of the in-flight states so it is reaped at most once.
    """
    container = build_container_for_session(self.db)
    outcome = reap_stale_batches(
        batches=container.batch_repository(self.db),
        reco_ledger=container.reconciliation_ledger(self.db),
        lease_seconds=_stale_batch_lease_seconds(),
    )
    return {
        "batches_reaped": outcome.batches_reaped,
        "events_failed": outcome.events_failed,
        "errors": outcome.errors,
        "reaped_batch_ids": outcome.reaped_batch_ids,
    }
