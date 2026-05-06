"""Celery tasks for context-graph ingestion workflows."""

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from application.use_cases.context_graph_jobs import (
    handle_apply_episode,
    handle_backfill_pot,
    handle_ingest_pr,
    handle_ingestion_agent_run,
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
    name="app.modules.context_graph.tasks.context_graph_ingest_pr",
    queue="context-graph-etl",
)
def context_graph_ingest_pr(
    self,
    pot_id: str,
    pr_number: int,
    is_live_bridge: bool = True,
    repo_name: str | None = None,
) -> dict:
    return handle_ingest_pr(
        self.db,
        pot_id,
        pr_number,
        is_live_bridge=is_live_bridge,
        repo_name=repo_name,
        build_container=build_container_for_session,
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_ingestion_agent_run",
    queue="context-graph-etl",
)
def context_graph_ingestion_agent_run(self, event_id: str) -> dict:
    return handle_ingestion_agent_run(
        self.db, event_id, build_container=build_container_for_session
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_apply_episode",
    queue="context-graph-etl",
)
def context_graph_apply_episode(self, pot_id: str, event_id: str, sequence: int) -> dict:
    return handle_apply_episode(
        self.db,
        pot_id,
        event_id,
        sequence,
        build_container=build_container_for_session,
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_sync_linear_project_source",
    queue="context-graph-etl",
)
def context_graph_sync_linear_project_source(self, project_source_id: str) -> dict:
    from integrations.adapters.outbound.linear.linear_sync import sync_linear_project_source

    return sync_linear_project_source(self.db, project_source_id)
