"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from adapters.inbound.http.deps import (
    get_container_or_503,
    get_db,
    require_api_key,
)
from application.use_cases.backfill_pot import backfill_pot_context
from application.use_cases.ingest_episode import ingest_episode
from application.use_cases.ingest_single_pr import ingest_single_pull_request
from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    get_pr_diff,
    get_pr_review_context,
    search_pot_context,
)
from bootstrap.container import ContextEngineContainer


class SyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_ids: Optional[list[str]] = Field(
        default=None,
        description="If set, only these pot IDs (must exist in CONTEXT_ENGINE_POTS).",
    )


class IngestPrRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int
    is_live_bridge: bool = True


class IngestEpisodeRequest(BaseModel):
    """Raw Graphiti episode (same fields as episodic.add_episode)."""

    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    name: str
    episode_body: str
    source_description: str
    reference_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event time for the episode (defaults to UTC now).",
    )


class ChangeHistoryQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    limit: int = 10
    repo_name: Optional[str] = None


class FileOwnersQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    file_path: str
    limit: int = 5
    repo_name: Optional[str] = None


class DecisionsQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    limit: int = 20
    repo_name: Optional[str] = None


class PrReviewContextQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int = Field(ge=1, description="GitHub pull request number")
    repo_name: Optional[str] = None


class PrDiffQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int = Field(ge=1, description="GitHub pull request number")
    file_path: Optional[str] = None
    limit: int = 30
    repo_name: Optional[str] = None


class SearchQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    query: str
    limit: int = 8
    node_labels: Optional[list[str]] = None
    repo_name: Optional[str] = None


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


def _require_pot_access(
    container: ContextEngineContainer, pot_id: str
) -> None:
    if container.pots.resolve_pot(pot_id) is None:
        raise HTTPException(status_code=404, detail="Unknown pot_id")


def _inline_sync(
    payload: Optional[SyncRequest],
    container: ContextEngineContainer,
    db: Session,
) -> dict[str, Any]:
    if not container.settings.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
        )
    mapping = payload.pot_ids if payload and payload.pot_ids else None
    results: list[dict[str, Any]] = []
    pots = container.pots
    if not hasattr(pots, "known_pot_ids"):
        raise HTTPException(
            status_code=500,
            detail="Pot resolution does not support listing IDs",
        )
    ids = mapping or pots.known_pot_ids()  # type: ignore[union-attr]
    for pid in ids:
        resolved = container.pots.resolve_pot(pid)
        primary = resolved.primary_repo() if resolved else None
        if not resolved or not primary:
            results.append(
                {
                    "status": "skipped",
                    "pot_id": pid,
                    "reason": "unknown_pot_id",
                }
            )
            continue
        out = backfill_pot_context(
            settings=container.settings,
            pots=container.pots,
            source=container.source_for_repo(primary.repo_name),
            ledger=container.ledger(db),
            episodic=container.episodic,
            structural=container.structural,
            pot_id=pid,
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
    resolved = container.pots.resolve_pot(body.pot_id)
    primary = resolved.primary_repo() if resolved else None
    if not resolved or not primary:
        raise HTTPException(status_code=404, detail="Unknown pot_id")
    return ingest_single_pull_request(
        settings=container.settings,
        pots=container.pots,
        source=container.source_for_repo(primary.repo_name),
        ledger=container.ledger(db),
        episodic=container.episodic,
        structural=container.structural,
        pot_id=body.pot_id,
        pr_number=body.pr_number,
        is_live_bridge=body.is_live_bridge,
    )


def create_context_router(
    *,
    require_auth: Callable[..., Any],
    get_container: Callable[..., ContextEngineContainer],
    get_db: Callable[..., Any],
    mutation_handlers: ContextMutationHandlers | None = None,
    enforce_pot_access: bool = False,
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

    @router.post(
        "/ingest",
        summary="Add episodic episode",
        description="Ingest a raw episode into Graphiti for the pot (group_id).",
    )
    def post_ingest_episode(
        body: IngestEpisodeRequest,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        if container.pots.resolve_pot(body.pot_id) is None:
            raise HTTPException(status_code=404, detail="Unknown pot_id")
        out = ingest_episode(
            container.episodic,
            body.pot_id,
            body.name,
            body.episode_body,
            body.source_description,
            body.reference_time,
        )
        if out.get("episode_uuid") is None:
            raise HTTPException(
                status_code=503,
                detail="Failed to add episode (Graphiti unavailable or ingestion error).",
            )
        return out

    @router.post("/query/change-history")
    def post_change_history(
        body: ChangeHistoryQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_change_history(
            container.structural,
            body.pot_id,
            function_name=body.function_name,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
        )

    @router.post("/query/file-owners")
    def post_file_owners(
        body: FileOwnersQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_file_owners(
            container.structural, body.pot_id, body.file_path, body.limit, repo_name=body.repo_name
        )

    @router.post("/query/decisions")
    def post_decisions(
        body: DecisionsQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_decisions(
            container.structural,
            body.pot_id,
            file_path=body.file_path,
            function_name=body.function_name,
            limit=body.limit,
            repo_name=body.repo_name,
        )

    @router.post("/query/pr-review-context")
    def post_pr_review_context(
        body: PrReviewContextQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_pr_review_context(
            container.structural, body.pot_id, body.pr_number, repo_name=body.repo_name
        )

    @router.post("/query/pr-diff")
    def post_pr_diff(
        body: PrDiffQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_pr_diff(
            container.structural,
            body.pot_id,
            body.pr_number,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
        )

    @router.post(
        "/query/search",
        summary="Semantic search (Graphiti episodic)",
    )
    def post_search(
        body: SearchQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return search_pot_context(
            container.episodic,
            body.pot_id,
            body.query,
            limit=body.limit,
            node_labels=body.node_labels,
            repo_name=body.repo_name,
        )

    return router


context_router = create_context_router(
    require_auth=require_api_key,
    get_container=get_container_or_503,
    get_db=get_db,
)
