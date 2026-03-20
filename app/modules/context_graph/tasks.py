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
# Max default-branch commits to ingest per project per backfill
BACKFILL_COMMIT_LIMIT = 100
# Cap comments/review comments per PR to avoid huge payloads
MAX_COMMENTS_PER_PR = 50
MAX_REVIEW_COMMENTS_PER_PR = 100


def _pr_to_payload(pr: Any) -> Dict[str, Any]:
    """Build a dict compatible with format_github_pr_episode from a PyGithub PullRequest.
    Enriches with issue comments, review comments, and commit messages when available.
    """
    payload = {
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
    try:
        comments = []
        for c in list(pr.get_issue_comments())[:MAX_COMMENTS_PER_PR]:
            comments.append({
                "body": c.body or "",
                "user": {"login": c.user.login} if c.user else {},
                "created_at": c.created_at.isoformat() if c.created_at else None,
            })
        payload["comments"] = comments
    except Exception as e:
        logger.debug("Context graph backfill: could not fetch issue comments for PR %s: %s", pr.number, e)
        payload["comments"] = []
    try:
        review_comments = []
        for c in list(pr.get_review_comments())[:MAX_REVIEW_COMMENTS_PER_PR]:
            review_comments.append({
                "body": c.body or "",
                "user": {"login": c.user.login} if c.user else {},
                "path": getattr(c, "path", None),
                "created_at": c.created_at.isoformat() if c.created_at else None,
            })
        payload["review_comments"] = review_comments
    except Exception as e:
        logger.debug("Context graph backfill: could not fetch review comments for PR %s: %s", pr.number, e)
        payload["review_comments"] = []
    try:
        commit_messages = []
        for c in pr.get_commits():
            msg = (c.commit.message or "").strip().split("\n")[0] if c.commit else ""
            author = (c.author.login if c.author else None) or (c.commit.author.name if c.commit and c.commit.author else "") or "?"
            commit_messages.append({"sha": c.sha[:12] if c.sha else "?", "message": msg, "author": author})
            if len(commit_messages) >= 50:
                break
        payload["commit_messages"] = commit_messages
    except Exception as e:
        logger.debug("Context graph backfill: could not fetch commits for PR %s: %s", pr.number, e)
        payload["commit_messages"] = []
    return payload


async def _backfill_async(
    task: BaseTask,
    project_id: str,
    pr_payloads: List[Dict[str, Any]],
    commit_payloads: List[Dict[str, Any]] | None = None,
    default_branch: str = "main",
) -> None:
    """Run inside run_async: ingest each PR, then each commit, and update sync_state.
    Commits after each ingest to avoid long transactions and connection timeouts.
    """
    now = datetime.utcnow()
    async with task.async_db() as db:
        try:
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
                    await db.commit()
                except Exception as e:
                    await db.rollback()
                    logger.warning(
                        "Context graph backfill: failed to ingest pr %s: %s",
                        pr.get("number"),
                        e,
                    )
            if commit_payloads:
                for commit in commit_payloads:
                    try:
                        sha = commit.get("sha", "")[:12] or "unknown"
                        await ingest_episode(
                            db,
                            project_id=project_id,
                            source_type="github_commit",
                            source_id=f"commit_{sha}",
                            payload=commit,
                            event_type=None,
                            branch_name=default_branch,
                        )
                        await db.commit()
                    except Exception as e:
                        await db.rollback()
                        logger.warning(
                            "Context graph backfill: failed to ingest commit %s: %s",
                            sha,
                            e,
                        )
            # Update sync_state for github_pr and github_commit
            await db.execute(
                update(ContextSyncState)
                .where(
                    ContextSyncState.project_id == project_id,
                    ContextSyncState.source_type == "github_pr",
                )
                .values(last_synced_at=now, status="idle", error=None, updated_at=now)
            )
            if commit_payloads:
                await db.execute(
                    update(ContextSyncState)
                    .where(
                        ContextSyncState.project_id == project_id,
                        ContextSyncState.source_type == "github_commit",
                    )
                    .values(last_synced_at=now, status="idle", error=None, updated_at=now)
                )
            await db.commit()
        except Exception:
            try:
                await db.rollback()
            except Exception as rb_e:
                logger.warning("Context graph backfill: rollback on error failed: %s", rb_e)
            raise


def _commit_to_payload(c: Any) -> Dict[str, Any]:
    """Build a dict for format_github_commit_episode from a PyGithub Commit."""
    sha = (getattr(c, "sha", "") or "")[:12] or "unknown"
    commit = getattr(c, "commit", None)
    message = ""
    author_name = "unknown"
    author_date = None
    if commit:
        message = (getattr(commit, "message", "") or "").strip().split("\n")[0] or "(no message)"
        author = getattr(commit, "author", None)
        if author:
            author_name = getattr(author, "name", None) or getattr(author, "email", None) or "unknown"
            author_date = getattr(author, "date", None)
            if author_date and hasattr(author_date, "isoformat"):
                author_date = author_date.isoformat()
    login = getattr(c, "author", None) and getattr(c.author, "login", None)
    return {
        "sha": sha,
        "message": message,
        "author": login or author_name,
        "commit": {
            "message": message,
            "author": {"name": author_name, "date": author_date},
        },
        "created_at": author_date,
    }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.modules.context_graph.tasks.context_graph_backfill_project",
)
def context_graph_backfill_project(self: BaseTask, project_id: str) -> None:
    """Backfill GitHub PRs (with comments, review comments, commit messages) and default-branch commits."""
    if not config_provider.get_context_graph_config().get("enabled"):
        return
    project = self.db.query(Project).filter(Project.id == project_id).first()
    if not project or not project.repo_name:
        logger.info("Context graph backfill: project %s not found or no repo_name", project_id)
        return
    repo_name = project.repo_name

    # Ensure sync_state rows exist for github_pr and github_commit
    for source_type in ("github_pr", "github_commit"):
        sync_row = (
            self.db.query(ContextSyncState)
            .filter(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == source_type,
            )
            .first()
        )
        if not sync_row:
            self.db.add(
                ContextSyncState(project_id=project_id, source_type=source_type, status="idle")
            )
    self.db.commit()

    pr_payloads: List[Dict[str, Any]] = []
    commit_payloads: List[Dict[str, Any]] = []
    default_branch = getattr(project, "branch_name", None) or "main"
    try:
        from app.modules.code_provider.github.github_service import GithubService

        gh = GithubService(self.db)
        github = gh.get_github_app_client(repo_name)
        repo = github.get_repo(repo_name)
        default_branch = getattr(repo, "default_branch", None) or default_branch

        # Fetch merged PRs with comments, review comments, and commit messages
        pulls = repo.get_pulls(state="closed")
        count = 0
        for pr in pulls:
            if not getattr(pr, "merged", False):
                continue
            pr_payloads.append(_pr_to_payload(pr))
            count += 1
            if count >= BACKFILL_BATCH_SIZE:
                break

        # Fetch recent commits on default branch
        try:
            for c in repo.get_commits(sha=default_branch)[:BACKFILL_COMMIT_LIMIT]:
                commit_payloads.append(_commit_to_payload(c))
        except Exception as e:
            logger.warning(
                "Context graph backfill: could not fetch commits for %s (branch %s): %s",
                repo_name,
                default_branch,
                e,
            )
    except Exception as e:
        logger.exception("Context graph backfill: failed to fetch GitHub data for %s: %s", repo_name, e)
        return

    if not pr_payloads and not commit_payloads:
        return

    self.run_async(
        _backfill_async(
            self,
            project_id,
            pr_payloads,
            commit_payloads=commit_payloads or None,
            default_branch=default_branch,
        )
    )


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
