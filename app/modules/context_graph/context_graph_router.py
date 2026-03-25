"""API router for context graph: trigger ETL sync from the UI."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.tasks import context_graph_backfill_project
from app.modules.projects.projects_service import ProjectService
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context-graph", tags=["context-graph"])


class SyncRequest(BaseModel):
    """Optional filter for syncing only selected projects."""

    project_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional list of project IDs to sync. If omitted, sync all eligible projects.",
    )


@router.post("/sync-all")
async def sync_all_sources(
    payload: Optional[SyncRequest] = None,
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
    selected_project_ids = set(payload.project_ids or []) if payload else set()
    if selected_project_ids:
        eligible = [p for p in eligible if p.get("id") in selected_project_ids]
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
