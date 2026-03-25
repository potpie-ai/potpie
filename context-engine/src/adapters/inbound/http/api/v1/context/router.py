"""Context graph HTTP API (standalone service)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from adapters.inbound.http.deps import (
    get_container_or_503,
    get_db,
    require_api_key,
)
from application.use_cases.backfill_project import backfill_project_context
from application.use_cases.ingest_single_pr import ingest_single_pull_request
from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    search_project_context,
)
from bootstrap.container import ContextEngineContainer

context_router = APIRouter()


class SyncRequest(BaseModel):
    project_ids: Optional[list[str]] = Field(
        default=None,
        description="If set, only these project IDs (must exist in CONTEXT_ENGINE_PROJECTS).",
    )


class IngestPrRequest(BaseModel):
    project_id: str
    pr_number: int
    is_live_bridge: bool = True


class ChangeHistoryQuery(BaseModel):
    project_id: str
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    limit: int = 10


class FileOwnersQuery(BaseModel):
    project_id: str
    file_path: str
    limit: int = 5


class DecisionsQuery(BaseModel):
    project_id: str
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    limit: int = 20


class SearchQuery(BaseModel):
    project_id: str
    query: str
    limit: int = 8
    node_labels: Optional[list[str]] = None


@context_router.post("/sync")
def post_sync(
    payload: Optional[SyncRequest] = None,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
    db: Session = Depends(get_db),
):
    if not container.settings.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Context graph is not enabled. Set CONTEXT_GRAPH_ENABLED=true.",
        )
    mapping = payload.project_ids if payload and payload.project_ids else None
    results = []
    projects = container.projects
    if not hasattr(projects, "known_project_ids"):
        raise HTTPException(500, detail="Project resolution does not support listing IDs")
    ids = mapping or projects.known_project_ids()  # type: ignore[union-attr]
    for pid in ids:
        resolved = container.projects.resolve(pid)
        if not resolved:
            results.append({"status": "skipped", "project_id": pid, "reason": "unknown_project_id"})
            continue
        out = backfill_project_context(
            settings=container.settings,
            projects=container.projects,
            source=container.source_for_repo(resolved.repo_name),
            ledger=container.ledger(db),
            episodic=container.episodic,
            structural=container.structural,
            project_id=pid,
        )
        results.append(out)
    return {"status": "success", "results": results}


@context_router.post("/ingest-pr")
def post_ingest_pr(
    body: IngestPrRequest,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
    db: Session = Depends(get_db),
):
    if not container.settings.is_enabled():
        raise HTTPException(status_code=503, detail="Context graph is not enabled.")
    resolved = container.projects.resolve(body.project_id)
    if not resolved:
        raise HTTPException(404, detail="Unknown project_id")
    return ingest_single_pull_request(
        settings=container.settings,
        projects=container.projects,
        source=container.source_for_repo(resolved.repo_name),
        ledger=container.ledger(db),
        episodic=container.episodic,
        structural=container.structural,
        project_id=body.project_id,
        pr_number=body.pr_number,
        is_live_bridge=body.is_live_bridge,
    )


@context_router.post("/query/change-history")
def post_change_history(
    body: ChangeHistoryQuery,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
):
    return get_change_history(
        container.structural,
        body.project_id,
        function_name=body.function_name,
        file_path=body.file_path,
        limit=body.limit,
    )


@context_router.post("/query/file-owners")
def post_file_owners(
    body: FileOwnersQuery,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
):
    return get_file_owners(container.structural, body.project_id, body.file_path, body.limit)


@context_router.post("/query/decisions")
def post_decisions(
    body: DecisionsQuery,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
):
    return get_decisions(
        container.structural,
        body.project_id,
        file_path=body.file_path,
        function_name=body.function_name,
        limit=body.limit,
    )


@context_router.post("/query/search")
def post_search(
    body: SearchQuery,
    _: None = Depends(require_api_key),
    container: ContextEngineContainer = Depends(get_container_or_503),
):
    return search_project_context(
        container.episodic,
        body.project_id,
        body.query,
        limit=body.limit,
        node_labels=body.node_labels,
    )
