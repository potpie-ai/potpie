"""Celery tasks for context-graph ingestion workflows."""

import time
from uuid import uuid4

from celery import Task

from app.celery.celery_app import celery_app
from app.core.config_provider import config_provider
from app.core.database import SessionLocal
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.context_graph.bridge_writer import write_bridges
from app.modules.context_graph.github_pr_fetcher import fetch_full_pr
from app.modules.context_graph.ingestion_service import ingest_pr
from app.modules.context_graph.models import ContextIngestionLog, ContextSyncState
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.logger import setup_logger

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


def _get_or_create_sync_state(db, project_id: str, source_type: str) -> ContextSyncState:
    state = (
        db.query(ContextSyncState)
        .filter(
            ContextSyncState.project_id == project_id,
            ContextSyncState.source_type == source_type,
        )
        .first()
    )
    if state:
        return state

    state = ContextSyncState(
        id=str(uuid4()),
        project_id=project_id,
        source_type=source_type,
        status="idle",
    )
    db.add(state)
    db.commit()
    db.refresh(state)
    return state


def _update_bridge_status(
    db,
    project_id: str,
    source_id: str,
    entity_key: str,
    bridge_result=None,
    error: str | None = None,
) -> None:
    """Persist bridge outcome into context_ingestion_log."""
    values: dict = {"entity_key": entity_key}
    if error:
        values["bridge_status"] = "failed"
        values["bridge_error"] = error[:2000]
        values["bridge_written"] = False
    elif bridge_result:
        values["bridge_status"] = "success"
        values["bridge_error"] = None
        values["bridge_written"] = True
        values["bridge_touched_by"] = bridge_result.touched_by
        values["bridge_modified_in"] = bridge_result.modified_in
        values["bridge_has_decision"] = bridge_result.has_decision
    else:
        values["bridge_status"] = "skipped"

    (
        db.query(ContextIngestionLog)
        .filter(
            ContextIngestionLog.project_id == project_id,
            ContextIngestionLog.source_type == "github_pr",
            ContextIngestionLog.source_id == source_id,
        )
        .update(values)
    )
    db.commit()


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_backfill_project",
    queue="context-graph-etl",
)
def context_graph_backfill_project(self, project_id: str) -> dict:
    """Backfill merged PR context for a project into Graphiti."""
    source_type = "github_pr"
    if not config_provider.get_context_graph_config().get("enabled"):
        return {
            "status": "skipped",
            "project_id": project_id,
            "reason": "context_graph_disabled",
        }

    project = ProjectService.get_project_by_id(self.db, project_id)
    if not project:
        logger.warning("Context graph backfill skipped: project not found %s", project_id)
        return {
            "status": "skipped",
            "project_id": project_id,
            "reason": "project_not_found",
        }
    if not project.repo_name:
        return {
            "status": "skipped",
            "project_id": project_id,
            "reason": "project_missing_repo_name",
        }

    sync_state = _get_or_create_sync_state(self.db, project_id, source_type)
    sync_state.status = "running"
    sync_state.error = None
    self.db.commit()

    ingested = 0
    skipped = 0
    failed = 0
    latest_merged_at = sync_state.last_synced_at

    try:
        provider = CodeProviderFactory.create_provider_with_fallback(project.repo_name)
        provider_client = provider.get_client()
        if provider_client is None:
            raise RuntimeError("Code provider client not available for PR backfill")

        repo = provider_client.get_repo(project.repo_name)
        pulls = repo.get_pulls(state="closed", sort="updated", direction="asc")

        for pr in pulls:
            if ingested + skipped >= 100:
                break
            if not pr.merged_at:
                continue
            if sync_state.last_synced_at and pr.merged_at <= sync_state.last_synced_at:
                continue

            source_id = f"pr_{pr.number}_merged"
            try:
                payload = fetch_full_pr(provider, project.repo_name, pr.number)
                result = ingest_pr(
                    db=self.db,
                    project_id=project_id,
                    repo_name=project.repo_name,
                    pr_data=payload["pr_data"],
                    commits=payload["commits"],
                    review_threads=payload["review_threads"],
                    linked_issues=payload["linked_issues"],
                    issue_comments=payload["issue_comments"],
                )

                bridge_result = write_bridges(
                    project_id=project_id,
                    pr_entity_key=result.pr_entity_key,
                    pr_number=pr.number,
                    repo_name=project.repo_name,
                    files_with_patches=payload["pr_data"].get("files", []),
                    review_threads=payload["review_threads"],
                    merged_at=pr.merged_at.isoformat() if pr.merged_at else None,
                    is_live=False,
                )
                _update_bridge_status(
                    self.db,
                    project_id,
                    source_id,
                    entity_key=result.pr_entity_key,
                    bridge_result=bridge_result,
                )
                ingested += 1
                latest_merged_at = pr.merged_at
                time.sleep(0.5)
            except Exception as exc:
                failed += 1
                _update_bridge_status(
                    self.db,
                    project_id,
                    source_id,
                    entity_key=f"github:pr:{project.repo_name}:{pr.number}",
                    error=str(exc),
                )
                logger.exception(
                    "Failed ingesting PR #%s for project %s",
                    pr.number,
                    project_id,
                )

        sync_state.status = "success"
        sync_state.last_synced_at = latest_merged_at
        sync_state.error = None
        self.db.commit()
        return {
            "status": "success",
            "project_id": project_id,
            "repo_name": project.repo_name,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "last_synced_at": latest_merged_at.isoformat() if latest_merged_at else None,
        }
    except Exception as exc:
        self.db.rollback()
        sync_state = _get_or_create_sync_state(self.db, project_id, source_type)
        sync_state.status = "error"
        sync_state.error = str(exc)
        self.db.commit()
        logger.exception("Context graph backfill failed for project %s", project_id)
        return {
            "status": "error",
            "project_id": project_id,
            "repo_name": project.repo_name,
            "error": str(exc),
            "ingested": ingested,
            "failed": failed,
        }


@celery_app.task(
    bind=True,
    base=ContextGraphTask,
    name="app.modules.context_graph.tasks.context_graph_ingest_pr",
    queue="context-graph-etl",
)
def context_graph_ingest_pr(self, project_id: str, pr_number: int) -> dict:
    """Ingest a single PR (typically webhook-driven)."""
    if not config_provider.get_context_graph_config().get("enabled"):
        return {
            "status": "skipped",
            "project_id": project_id,
            "pr_number": pr_number,
            "reason": "context_graph_disabled",
        }

    project = ProjectService.get_project_by_id(self.db, project_id)
    if not project or not project.repo_name:
        return {
            "status": "skipped",
            "project_id": project_id,
            "pr_number": pr_number,
            "reason": "project_not_found_or_missing_repo",
        }

    source_id = f"pr_{pr_number}_merged"
    try:
        provider = CodeProviderFactory.create_provider_with_fallback(project.repo_name)
        payload = fetch_full_pr(provider, project.repo_name, pr_number)
        result = ingest_pr(
            db=self.db,
            project_id=project_id,
            repo_name=project.repo_name,
            pr_data=payload["pr_data"],
            commits=payload["commits"],
            review_threads=payload["review_threads"],
            linked_issues=payload["linked_issues"],
            issue_comments=payload["issue_comments"],
        )

        bridge_result = write_bridges(
            project_id=project_id,
            pr_entity_key=result.pr_entity_key,
            pr_number=pr_number,
            repo_name=project.repo_name,
            files_with_patches=payload["pr_data"].get("files", []),
            review_threads=payload["review_threads"],
            merged_at=payload["pr_data"].get("merged_at"),
            is_live=True,
        )
        _update_bridge_status(
            self.db,
            project_id,
            source_id,
            entity_key=result.pr_entity_key,
            bridge_result=bridge_result,
        )
        return {
            "status": "success",
            "project_id": project_id,
            "repo_name": project.repo_name,
            "pr_number": pr_number,
            "bridges": bridge_result.as_dict(),
        }
    except Exception as exc:
        _update_bridge_status(
            self.db,
            project_id,
            source_id,
            entity_key=f"github:pr:{project.repo_name}:{pr_number}",
            error=str(exc),
        )
        logger.exception(
            "Single PR ingest failed for project=%s pr=%s",
            project_id,
            pr_number,
        )
        return {
            "status": "error",
            "project_id": project_id,
            "pr_number": pr_number,
            "error": str(exc),
        }
