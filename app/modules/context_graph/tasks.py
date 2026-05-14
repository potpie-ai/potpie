"""Celery tasks for context-graph ingestion workflows."""

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from application.use_cases.context_graph_jobs import (
    handle_backfill_pot,
    handle_process_batch,
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


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_backfill_pot",
    queue="context-graph-etl",
)
def context_graph_backfill_pot(
    self, pot_id: str, target_repo_name: str | None = None
) -> dict:
    return handle_backfill_pot(
        self.db,
        pot_id,
        target_repo_name=target_repo_name,
        build_container=build_container_for_session,
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_sync_linear_pot_source",
    queue="context-graph-etl",
)
def context_graph_sync_linear_pot_source(self, pot_source_id: str) -> dict:
    """Backfill a Linear ``context_graph_pot_sources`` row into the graph.

    Invoked when a Linear team is attached to a pot via the context-engine
    sources API; enumerates issues for that team and feeds each through
    ``LinearConnector.normalize_webhook`` → ``IngestionSubmissionService``
    so backfill and live webhooks share the same admission path.
    """
    from integrations.application.backfill_linear_source import (
        backfill_linear_source,
    )

    return backfill_linear_source(self.db, pot_source_id)


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_process_batch",
    queue="context-graph-etl",
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
