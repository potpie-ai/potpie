"""API router for context graph: trigger ETL sync from the UI."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.tasks import context_graph_backfill_project
from app.modules.projects.projects_service import ProjectService
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context-graph", tags=["context-graph"])


@router.post("/sync-all")
async def sync_all_sources(
    user: dict = Depends(AuthService.check_auth),
    db=Depends(get_db),
):
    """Enqueue context graph backfill for all of the authenticated user's projects that have a repo (GitHub)."""
    if not config_provider.get_context_graph_config().get("enabled"):
        raise HTTPException(
            status_code=503,
            detail="Context graph is not enabled. Set CONTEXT_GRAPH_ENABLED=true.",
        )
    user_id = user["user_id"]
    project_service = ProjectService(db)
    project_list = await project_service.list_projects(user_id)
    eligible = [
        p for p in project_list
        if p.get("repo_name") and (p.get("status") or "").lower() == "ready"
    ]
    enqueued = 0
    for p in eligible:
        try:
            context_graph_backfill_project.delay(p["id"])
            enqueued += 1
        except Exception as e:
            logger.warning("Failed to enqueue backfill for project %s: %s", p.get("id"), e)
    return {
        "status": "success",
        "message": f"Enqueued context graph sync for {enqueued} project(s).",
        "enqueued": enqueued,
        "total_eligible": len(eligible),
    }
