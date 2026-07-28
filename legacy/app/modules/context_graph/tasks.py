"""Celery task for context-graph batch processing.

One background job exists (see ``ContextGraphJobQueuePort``): a per-batch
processor. Every admitted ingestion event lands in a batch; ``enqueue_batch``
fires this task, which claims the batch and runs the reconciliation agent
over it. Backfill and PR ingest are no longer standalone jobs — they flow
through event admission into this same path.
"""

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_ingestion_server_for_session
from potpie_context_engine.application.use_cases.context_graph_jobs import (
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
    name="app.modules.context_graph.tasks.context_graph_process_batch",
    queue="context-graph-etl",
)
def context_graph_process_batch(self, batch_id: str) -> dict:
    return handle_process_batch(
        self.db,
        batch_id,
        build_ingestion_server=build_ingestion_server_for_session,
    )
