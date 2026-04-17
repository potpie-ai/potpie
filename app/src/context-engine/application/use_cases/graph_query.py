"""Single entrypoint for all context-graph read queries (HTTP + CLI + MCP)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    get_pr_diff,
    get_pr_review_context,
    get_project_graph,
    search_pot_context,
)
from application.use_cases.resolve_context import resolve_context
from bootstrap.container import ContextEngineContainer
from domain.agent_context_port import bundle_to_agent_envelope
from domain.graph_query import GraphQueryKind
from domain.intelligence_models import (
    ArtifactRef,
    ContextResolutionRequest,
    ContextScope,
)

logger = logging.getLogger(__name__)


class GraphQueryArtifact(BaseModel):
    """Optional artifact hint for ``resolve_context``."""

    kind: str = Field(description="e.g. pr, issue")
    identifier: str = Field(description="e.g. PR number")


class GraphQueryScope(BaseModel):
    """Optional scope hint for ``resolve_context``."""

    repo_name: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    symbol: Optional[str] = None
    pr_number: Optional[int] = Field(default=None, ge=1)
    services: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    environment: Optional[str] = None
    user: Optional[str] = None
    source_refs: list[str] = Field(default_factory=list)


class GraphQueryRequest(BaseModel):
    """Unified request: set ``kind`` and the fields relevant to that mode."""

    model_config = {"populate_by_name": True}

    kind: GraphQueryKind
    pot_id: Optional[str] = Field(
        default=None,
        description="Pot scope (Graphiti group_id; structural group_id / repoId).",
    )
    query: Optional[str] = Field(
        default=None,
        description="Natural-language query for semantic_search and resolve_context.",
    )
    limit: int = Field(default=10, ge=1, le=200)
    repo_name: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    pr_number: Optional[int] = Field(default=None, ge=1)
    node_labels: Optional[list[str]] = None
    include: list[str] = Field(default_factory=list)
    source_description: Optional[str] = None
    include_invalidated: bool = Field(
        default=False,
        description="For semantic_search: include Graphiti edges marked invalid (superseded). Ignored when as_of is set.",
    )
    as_of: Optional[datetime] = Field(
        default=None,
        description="For semantic_search: restrict to edges valid at this instant.",
    )
    consumer_hint: Optional[str] = None
    intent: Optional[str] = None
    exclude: list[str] = Field(default_factory=list)
    mode: str = "fast"
    source_policy: str = "references_only"
    timeout_ms: int = Field(default=4000, ge=500, le=30000)
    artifact: Optional[GraphQueryArtifact] = None
    scope: Optional[GraphQueryScope] = None

    def scope_id(self) -> str:
        return (self.pot_id or "").strip()

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> GraphQueryRequest:
        k = self.kind
        sid = self.scope_id()
        if k in (
            GraphQueryKind.SEMANTIC_SEARCH,
            GraphQueryKind.CHANGE_HISTORY,
            GraphQueryKind.FILE_OWNERS,
            GraphQueryKind.DECISIONS,
            GraphQueryKind.PR_REVIEW_CONTEXT,
            GraphQueryKind.PR_DIFF,
        ):
            if not sid:
                raise ValueError("pot_id is required for this kind")
        if k in (GraphQueryKind.PROJECT_GRAPH, GraphQueryKind.RESOLVE_CONTEXT):
            if not sid:
                raise ValueError("pot_id is required for this kind")
        if k == GraphQueryKind.SEMANTIC_SEARCH and not (
            self.query and self.query.strip()
        ):
            raise ValueError("query is required for semantic_search")
        if k == GraphQueryKind.FILE_OWNERS and not (
            self.file_path and self.file_path.strip()
        ):
            raise ValueError("file_path is required for file_owners")
        if (
            k in (GraphQueryKind.PR_REVIEW_CONTEXT, GraphQueryKind.PR_DIFF)
            and self.pr_number is None
        ):
            raise ValueError("pr_number is required for this kind")
        if k == GraphQueryKind.RESOLVE_CONTEXT and not (
            self.query and self.query.strip()
        ):
            raise ValueError("query is required for resolve_context")
        return self


def _dispatch_sync(
    container: ContextEngineContainer,
    body: GraphQueryRequest,
) -> dict[str, Any]:
    """Execute all graph query kinds except ``resolve_context``."""
    k = body.kind
    sid = body.scope_id()
    if k == GraphQueryKind.SEMANTIC_SEARCH:
        rows = search_pot_context(
            container.episodic,
            sid,
            body.query or "",
            limit=body.limit,
            node_labels=body.node_labels,
            repo_name=body.repo_name,
            source_description=body.source_description,
            include_invalidated=body.include_invalidated,
            as_of=body.as_of,
        )
        return {"kind": k.value, "result": rows}

    if k == GraphQueryKind.CHANGE_HISTORY:
        rows = get_change_history(
            container.structural,
            sid,
            function_name=body.function_name,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
            pr_number=body.pr_number,
        )
        return {"kind": k.value, "result": rows}

    if k == GraphQueryKind.FILE_OWNERS:
        rows = get_file_owners(
            container.structural,
            sid,
            body.file_path or "",
            limit=min(body.limit, 50),
            repo_name=body.repo_name,
        )
        return {"kind": k.value, "result": rows}

    if k == GraphQueryKind.DECISIONS:
        rows = get_decisions(
            container.structural,
            sid,
            file_path=body.file_path,
            function_name=body.function_name,
            limit=body.limit,
            repo_name=body.repo_name,
            pr_number=body.pr_number,
        )
        return {"kind": k.value, "result": rows}

    if k == GraphQueryKind.PR_REVIEW_CONTEXT:
        out = get_pr_review_context(
            container.structural,
            sid,
            body.pr_number or 0,
            repo_name=body.repo_name,
        )
        return {"kind": k.value, "result": out}

    if k == GraphQueryKind.PR_DIFF:
        rows = get_pr_diff(
            container.structural,
            sid,
            body.pr_number or 0,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
        )
        return {"kind": k.value, "result": rows}

    if k == GraphQueryKind.PROJECT_GRAPH:
        out = get_project_graph(
            container.structural,
            sid,
            pr_number=body.pr_number,
            limit=min(body.limit, 50),
            scope={
                "repo_name": body.repo_name,
                "services": body.scope.services if body.scope else [],
                "features": body.scope.features if body.scope else [],
                "environment": body.scope.environment if body.scope else None,
                "user": body.scope.user if body.scope else None,
            },
            include=body.include,
        )
        return {"kind": k.value, "result": out}

    raise ValueError(f"unsupported kind: {k}")


def run_graph_query_sync(
    container: ContextEngineContainer,
    body: GraphQueryRequest,
) -> dict[str, Any]:
    """CLI / MCP: sync API; ``resolve_context`` uses ``asyncio.run`` internally."""
    if body.kind == GraphQueryKind.RESOLVE_CONTEXT:
        return asyncio.run(_resolve_context_async(container, body))
    return _dispatch_sync(container, body)


async def run_graph_query_async(
    container: ContextEngineContainer,
    body: GraphQueryRequest,
) -> dict[str, Any]:
    """HTTP: async path so ``resolve_context`` does not nest ``asyncio.run`` under FastAPI."""
    if body.kind == GraphQueryKind.RESOLVE_CONTEXT:
        return await _resolve_context_async(container, body)
    return _dispatch_sync(container, body)


async def _resolve_context_async(
    container: ContextEngineContainer,
    body: GraphQueryRequest,
) -> dict[str, Any]:
    sid = body.scope_id()
    if container.resolution_service is None:
        return {
            "kind": GraphQueryKind.RESOLVE_CONTEXT.value,
            "result": None,
            "error": "resolution_service_unavailable",
        }
    art = (
        ArtifactRef(kind=body.artifact.kind, identifier=body.artifact.identifier)
        if body.artifact
        else None
    )
    scope = None
    if body.scope:
        scope = ContextScope(
            repo_name=body.scope.repo_name,
            file_path=body.scope.file_path,
            function_name=body.scope.function_name,
            symbol=body.scope.symbol,
            pr_number=body.scope.pr_number,
            services=list(body.scope.services),
            features=list(body.scope.features),
            environment=body.scope.environment,
            user=body.scope.user,
            source_refs=body.scope.source_refs,
        )
    req = ContextResolutionRequest(
        pot_id=sid,
        query=body.query or "",
        consumer_hint=body.consumer_hint,
        artifact_ref=art,
        scope=scope,
        intent=body.intent,
        include=body.include,
        exclude=body.exclude,
        mode=body.mode,
        source_policy=body.source_policy,
        timeout_ms=body.timeout_ms,
    )
    bundle = await resolve_context(container.resolution_service, req)
    return {
        "kind": GraphQueryKind.RESOLVE_CONTEXT.value,
        "result": bundle_to_agent_envelope(bundle),
    }
