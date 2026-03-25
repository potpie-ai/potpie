"""Celery tasks for context-graph ingestion workflows."""

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from app.modules.utils.logger import setup_logger
from application.use_cases.backfill_project import backfill_project_context
from application.use_cases.ingest_single_pr import ingest_single_pull_request

logger = setup_logger(__name__)


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
    name="app.modules.context_graph.tasks.context_graph_backfill_project",
    queue="context-graph-etl",
)
def context_graph_backfill_project(self, project_id: str) -> dict:
    container = build_container_for_session(self.db)
    resolved = container.projects.resolve(project_id)
    if not resolved:
        logger.warning("Context graph backfill skipped: project not found %s", project_id)
        return {
            "status": "skipped",
            "project_id": project_id,
            "reason": "project_not_found",
        }
    source = container.source_for_repo(resolved.repo_name)
    return backfill_project_context(
        settings=container.settings,
        projects=container.projects,
        source=source,
        ledger=container.ledger(self.db),
        episodic=container.episodic,
        structural=container.structural,
        project_id=project_id,
    )


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_ingest_pr",
    queue="context-graph-etl",
)
def context_graph_ingest_pr(
    self, project_id: str, pr_number: int, is_live_bridge: bool = True
) -> dict:
    container = build_container_for_session(self.db)
    resolved = container.projects.resolve(project_id)
    if not resolved:
        return {
            "status": "skipped",
            "project_id": project_id,
            "pr_number": pr_number,
            "reason": "project_not_found_or_missing_repo",
        }
    source = container.source_for_repo(resolved.repo_name)
    return ingest_single_pull_request(
        settings=container.settings,
        projects=container.projects,
        source=source,
        ledger=container.ledger(self.db),
        episodic=container.episodic,
        structural=container.structural,
        project_id=project_id,
        pr_number=pr_number,
        is_live_bridge=is_live_bridge,
    )
