"""Celery tasks for context-graph ingestion workflows."""

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from app.modules.utils.logger import setup_logger
from application.use_cases.backfill_pot import backfill_pot_context
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
    name="app.modules.context_graph.tasks.context_graph_backfill_pot",
    queue="context-graph-etl",
)
def context_graph_backfill_pot(self, pot_id: str) -> dict:
    container = build_container_for_session(self.db)
    resolved = container.pots.resolve_pot(pot_id)
    if not resolved:
        logger.warning("Context graph backfill skipped: pot not found %s", pot_id)
        return {"status": "skipped", "pot_id": pot_id, "reason": "pot_not_found"}
    if not resolved.repos:
        logger.warning("Context graph backfill skipped: no repos for pot %s", pot_id)
        return {"status": "skipped", "pot_id": pot_id, "reason": "no_repos"}
    github_repos = [r for r in resolved.repos if r.provider == "github"]
    if not github_repos:
        logger.warning(
            "Context graph backfill skipped: no github repos for pot %s", pot_id
        )
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "reason": "no_github_repos",
        }
    merged: dict = {
        "status": "success",
        "pot_id": pot_id,
        "repos": [],
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
    }
    for rr in github_repos:
        source = container.source_for_repo(rr.repo_name)
        one = backfill_pot_context(
            settings=container.settings,
            pots=container.pots,
            source=source,
            ledger=container.ledger(self.db),
            episodic=container.episodic,
            structural=container.structural,
            pot_id=pot_id,
            target_repo=rr,
        )
        merged["repos"].append({"repo_name": rr.repo_name, "result": one})
        if one.get("status") == "success":
            merged["ingested"] += int(one.get("ingested") or 0)
            merged["skipped"] += int(one.get("skipped") or 0)
            merged["failed"] += int(one.get("failed") or 0)
        elif one.get("status") == "error":
            merged["failed"] += 1
            merged.setdefault("errors", []).append(
                {"repo_name": rr.repo_name, "error": one.get("error")}
            )
    if merged.get("errors"):
        merged["status"] = "partial_error" if merged["ingested"] > 0 else "error"
    return merged


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
    container = build_container_for_session(self.db)
    resolved = container.pots.resolve_pot(pot_id)
    if not resolved or not resolved.repos:
        logger.warning(
            "PR ingest skipped: pot %s not found or no repos (pr=%s)",
            pot_id, pr_number,
        )
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "reason": "pot_not_found_or_missing_repo",
        }
    from application.use_cases.ingest_single_pr import pick_github_repo_for_pot

    rr = pick_github_repo_for_pot(resolved.repos, repo_name)
    if not rr:
        logger.warning(
            "PR ingest skipped: no github repo matching %r for pot %s (pr=%s)",
            repo_name, pot_id, pr_number,
        )
        return {
            "status": "skipped",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "reason": "no_github_repo",
        }
    source = container.source_for_repo(rr.repo_name)
    return ingest_single_pull_request(
        settings=container.settings,
        pots=container.pots,
        source=source,
        ledger=container.ledger(self.db),
        episodic=container.episodic,
        structural=container.structural,
        pot_id=pot_id,
        pr_number=pr_number,
        is_live_bridge=is_live_bridge,
        repo_name=rr.repo_name,
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
