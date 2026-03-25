"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

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


class ContextMutationHandlers(Protocol):
    """Host-provided mutations (e.g. Celery enqueue) instead of inline use cases."""

    def handle_sync(
        self,
        payload: Optional[SyncRequest],
        container: ContextEngineContainer,
        db: Session,
    ) -> dict[str, Any]:
        ...

    def handle_ingest_pr(
        self,
        body: IngestPrRequest,
        container: ContextEngineContainer,
        db: Session,
    ) -> dict[str, Any]:
        ...


def _require_project_access(
    container: ContextEngineContainer, project_id: str
) -> None:
    if container.projects.resolve(project_id) is None:
        raise HTTPException(status_code=404, detail="Unknown project_id")


def _inline_sync(
    payload: Optional[SyncRequest],
    container: ContextEngineContainer,
    db: Session,
) -> dict[str, Any]:
    if not container.settings.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Context graph is not enabled. Set CONTEXT_GRAPH_ENABLED=true.",
        )
    mapping = payload.project_ids if payload and payload.project_ids else None
    results: list[dict[str, Any]] = []
    projects = container.projects
    if not hasattr(projects, "known_project_ids"):
        raise HTTPException(
            status_code=500,
            detail="Project resolution does not support listing IDs",
        )
    ids = mapping or projects.known_project_ids()  # type: ignore[union-attr]
    for pid in ids:
        resolved = container.projects.resolve(pid)
        if not resolved:
            results.append(
                {
                    "status": "skipped",
                    "project_id": pid,
                    "reason": "unknown_project_id",
                }
            )
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


def _inline_ingest_pr(
    body: IngestPrRequest,
    container: ContextEngineContainer,
    db: Session,
) -> dict[str, Any]:
    if not container.settings.is_enabled():
        raise HTTPException(
            status_code=503, detail="Context graph is not enabled."
        )
    resolved = container.projects.resolve(body.project_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Unknown project_id")
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


def create_context_router(
    *,
    require_auth: Callable[..., Any],
    get_container: Callable[..., ContextEngineContainer],
    get_db: Callable[..., Any],
    mutation_handlers: ContextMutationHandlers | None = None,
    enforce_project_access: bool = False,
) -> APIRouter:
    """Build context API routes with injected FastAPI dependencies."""

    router = APIRouter()

    @router.post("/sync")
    def post_sync(
        payload: Optional[SyncRequest] = None,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        if mutation_handlers is not None:
            return mutation_handlers.handle_sync(payload, container, db)
        return _inline_sync(payload, container, db)

    @router.post("/ingest-pr")
    def post_ingest_pr(
        body: IngestPrRequest,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        if mutation_handlers is not None:
            return mutation_handlers.handle_ingest_pr(body, container, db)
        return _inline_ingest_pr(body, container, db)

    @router.post("/query/change-history")
    def post_change_history(
        body: ChangeHistoryQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_project_access:
            _require_project_access(container, body.project_id)
        return get_change_history(
            container.structural,
            body.project_id,
            function_name=body.function_name,
            file_path=body.file_path,
            limit=body.limit,
        )

    @router.post("/query/file-owners")
    def post_file_owners(
        body: FileOwnersQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_project_access:
            _require_project_access(container, body.project_id)
        return get_file_owners(
            container.structural, body.project_id, body.file_path, body.limit
        )

    @router.post("/query/decisions")
    def post_decisions(
        body: DecisionsQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_project_access:
            _require_project_access(container, body.project_id)
        return get_decisions(
            container.structural,
            body.project_id,
            file_path=body.file_path,
            function_name=body.function_name,
            limit=body.limit,
        )

    @router.post("/query/search")
    def post_search(
        body: SearchQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_project_access:
            _require_project_access(container, body.project_id)
        return search_project_context(
            container.episodic,
            body.project_id,
            body.query,
            limit=body.limit,
            node_labels=body.node_labels,
        )

    return router


context_router = create_context_router(
    require_auth=require_api_key,
    get_container=get_container_or_503,
    get_db=get_db,
)
