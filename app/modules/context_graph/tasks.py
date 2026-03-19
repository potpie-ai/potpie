"""Celery tasks for context graph ETL and webhook ingestion."""

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import update

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.core.config_provider import config_provider
from app.modules.context_graph.ingestion_service import ingest_episode
from app.modules.context_graph.models import ContextIngestionLog, ContextSyncState
from app.modules.projects.projects_model import Project
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Max PRs to process per backfill run (resume on next run via cursor)
BACKFILL_BATCH_SIZE = 100


def _pr_to_payload(pr: Any) -> Dict[str, Any]:
    """Build a dict compatible with format_github_pr_episode from a PyGithub PullRequest."""
    return {
        "number": pr.number,
        "title": pr.title or "",
        "body": pr.body or "",
        "state": pr.state,
        "created_at": pr.created_at.isoformat() if pr.created_at else None,
        "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "head_branch": pr.head.ref if pr.head else "",
        "base_branch": pr.base.ref if pr.base else "",
        "url": pr.html_url or "",
        "author": pr.user.login if pr.user else "unknown",
        "files": [],  # Optional; omit for backfill to avoid extra API calls
    }


async def _backfill_async(
    task: BaseTask,
    project_id: str,
    pr_payloads: List[Dict[str, Any]],
) -> None:
    """Run inside run_async: ingest each PR and update sync_state."""
    if not pr_payloads:
        return
    async with task.async_db() as db:
        for pr in pr_payloads:
            try:
                await ingest_episode(
                    db,
                    project_id=project_id,
                    source_type="github_pr",
                    source_id=f"pr_{pr['number']}_merged",
                    payload=pr,
                    event_type="merged",
                )
            except Exception as e:
                logger.warning(
                    "Context graph backfill: failed to ingest pr %s: %s",
                    pr.get("number"),
                    e,
                )
        # Update sync_state cursor and status
        now = datetime.utcnow()
        await db.execute(
            update(ContextSyncState)
            .where(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == "github_pr",
            )
            .values(last_synced_at=now, status="idle", error=None, updated_at=now)
        )
        await db.commit()


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.modules.context_graph.tasks.context_graph_backfill_project",
)
def context_graph_backfill_project(self: BaseTask, project_id: str) -> None:
    """Backfill GitHub PRs for one project into the context graph."""
    if not config_provider.get_context_graph_config().get("enabled"):
        return
    project = self.db.query(Project).filter(Project.id == project_id).first()
    if not project or not project.repo_name:
        logger.info("Context graph backfill: project %s not found or no repo_name", project_id)
        return
    repo_name = project.repo_name

    # Ensure sync_state row exists (so we can update it in async part)
    sync_row = (
        self.db.query(ContextSyncState)
        .filter(ContextSyncState.project_id == project_id, ContextSyncState.source_type == "github_pr")
        .first()
    )
    if not sync_row:
        self.db.add(
            ContextSyncState(project_id=project_id, source_type="github_pr", status="idle")
        )
        self.db.commit()

    # Sync: fetch merged PRs (blocking I/O before run_async)
    pr_payloads: List[Dict[str, Any]] = []
    try:
        from app.modules.code_provider.github.github_service import GithubService

        gh = GithubService(self.db)
        github = gh.get_github_app_client(repo_name)
        repo = github.get_repo(repo_name)
        pulls = repo.get_pulls(state="closed")
        count = 0
        for pr in pulls:
            if not getattr(pr, "merged", False):
                continue
            pr_payloads.append(_pr_to_payload(pr))
            count += 1
            if count >= BACKFILL_BATCH_SIZE:
                break
    except Exception as e:
        logger.exception("Context graph backfill: failed to fetch PRs for %s: %s", repo_name, e)
        return

    if not pr_payloads:
        return

    self.run_async(_backfill_async(self, project_id, pr_payloads))


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.modules.context_graph.tasks.ingest_pr_from_webhook",
)
def ingest_pr_from_webhook(self: BaseTask, project_id: str, pr_payload: Dict[str, Any]) -> None:
    """Ingest a single PR from a webhook (e.g. PR merged)."""

    async def _do_ingest() -> None:
        async with self.async_db() as db:
            pr_number = pr_payload.get("number")
            if pr_number is None:
                pr_number = 0
            else:
                pr_number = int(pr_number)
            if not pr_number:
                logger.warning("ingest_pr_from_webhook: no number in payload")
                return
            source_id = f"pr_{pr_number}_merged"
            # Normalize webhook payload to our shape (head/base refs, author)
            payload = dict(pr_payload)
            if "head" in payload and isinstance(payload["head"], dict):
                payload["head_branch"] = payload["head"].get("ref", payload.get("head_branch"))
            if "base" in payload and isinstance(payload["base"], dict):
                payload["base_branch"] = payload["base"].get("ref", payload.get("base_branch"))
            if "user" in payload and isinstance(payload["user"], dict):
                payload["author"] = payload["user"].get("login", payload.get("author"))
            await ingest_episode(
                db,
                project_id=project_id,
                source_type="github_pr",
                source_id=source_id,
                payload=payload,
                event_type="merged",
            )

    if not config_provider.get_context_graph_config().get("enabled"):
        return
    self.run_async(_do_ingest())
