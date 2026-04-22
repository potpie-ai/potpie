"""Unified graph query request model."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContextGraphGoal(StrEnum):
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    NEIGHBORHOOD = "neighborhood"
    TIMELINE = "timeline"
    AGGREGATE = "aggregate"


class ContextGraphStrategy(StrEnum):
    AUTO = "auto"
    SEMANTIC = "semantic"
    EXACT = "exact"
    HYBRID = "hybrid"
    TRAVERSAL = "traversal"
    TEMPORAL = "temporal"


class ContextGraphScope(BaseModel):
    model_config = {"populate_by_name": True}

    repo_name: str | None = None
    branch: str | None = None
    file_path: str | None = None
    function_name: str | None = None
    symbol: str | None = None
    pr_number: int | None = Field(default=None, ge=1)
    services: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    environment: str | None = None
    ticket_ids: list[str] = Field(default_factory=list)
    user: str | None = None
    source_refs: list[str] = Field(default_factory=list)


class ContextGraphArtifact(BaseModel):
    kind: str
    identifier: str


class ContextGraphBudget(BaseModel):
    max_items: int = Field(default=12, ge=1, le=50)
    max_tokens: int | None = Field(default=None, ge=256, le=200000)
    timeout_ms: int = Field(default=4000, ge=500, le=30000)
    freshness: str = "prefer_fresh"


class ContextGraphQuery(BaseModel):
    """Minimal, expressive graph read request used by the application layer."""

    model_config = {"populate_by_name": True}

    pot_id: str
    query: str | None = None
    goal: ContextGraphGoal = ContextGraphGoal.RETRIEVE
    strategy: ContextGraphStrategy = ContextGraphStrategy.AUTO
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    scope: ContextGraphScope = Field(default_factory=ContextGraphScope)
    node_labels: list[str] = Field(default_factory=list)
    source_descriptions: list[str] = Field(default_factory=list)
    episode_uuids: list[str] = Field(default_factory=list)
    as_of: datetime | None = None
    include_invalidated: bool = False
    limit: int = Field(default=12, ge=1, le=200)
    consumer_hint: str | None = None
    intent: str | None = None
    source_policy: str = "references_only"
    artifact: ContextGraphArtifact | None = None
    budget: ContextGraphBudget = Field(default_factory=ContextGraphBudget)


class ContextGraphResult(BaseModel):
    """Normalized graph query result."""

    kind: str
    goal: str
    strategy: str
    result: Any = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


def preset_semantic_search(
    *,
    pot_id: str,
    query: str,
    limit: int = 12,
    repo_name: str | None = None,
    node_labels: list[str] | None = None,
    source_description: str | None = None,
    include_invalidated: bool = False,
    as_of: datetime | None = None,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        query=query,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.SEMANTIC,
        include=["semantic_search"],
        scope=ContextGraphScope(repo_name=repo_name),
        node_labels=list(node_labels or []),
        source_descriptions=[source_description] if source_description else [],
        include_invalidated=include_invalidated,
        as_of=as_of,
        limit=limit,
    )


def preset_change_history(
    *,
    pot_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    repo_name: str | None = None,
    pr_number: int | None = None,
    limit: int = 10,
    as_of: datetime | None = None,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.TIMELINE,
        strategy=ContextGraphStrategy.TEMPORAL,
        include=["change_history"],
        scope=ContextGraphScope(
            repo_name=repo_name,
            file_path=file_path,
            function_name=function_name,
            pr_number=pr_number,
        ),
        limit=limit,
        as_of=as_of,
    )


def preset_file_owners(
    *,
    pot_id: str,
    file_path: str,
    repo_name: str | None = None,
    limit: int = 5,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.AGGREGATE,
        strategy=ContextGraphStrategy.EXACT,
        include=["owners"],
        scope=ContextGraphScope(repo_name=repo_name, file_path=file_path),
        limit=limit,
    )


def preset_decisions(
    *,
    pot_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    repo_name: str | None = None,
    pr_number: int | None = None,
    limit: int = 20,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.EXACT,
        include=["decisions"],
        scope=ContextGraphScope(
            repo_name=repo_name,
            file_path=file_path,
            function_name=function_name,
            pr_number=pr_number,
        ),
        limit=limit,
    )


def preset_pr_review_context(
    *,
    pot_id: str,
    pr_number: int,
    repo_name: str | None = None,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.EXACT,
        include=["pr_review_context"],
        scope=ContextGraphScope(repo_name=repo_name, pr_number=pr_number),
    )


def preset_pr_diff(
    *,
    pot_id: str,
    pr_number: int,
    file_path: str | None = None,
    repo_name: str | None = None,
    limit: int = 30,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.EXACT,
        include=["pr_diff"],
        scope=ContextGraphScope(
            repo_name=repo_name,
            file_path=file_path,
            pr_number=pr_number,
        ),
        limit=limit,
    )


def preset_project_graph(
    *,
    pot_id: str,
    repo_name: str | None = None,
    pr_number: int | None = None,
    services: list[str] | None = None,
    features: list[str] | None = None,
    environment: str | None = None,
    user: str | None = None,
    include: list[str] | None = None,
    limit: int = 12,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.NEIGHBORHOOD,
        strategy=ContextGraphStrategy.TRAVERSAL,
        include=list(include or []),
        scope=ContextGraphScope(
            repo_name=repo_name,
            pr_number=pr_number,
            services=list(services or []),
            features=list(features or []),
            environment=environment,
            user=user,
        ),
        limit=limit,
    )


def preset_graph_overview(
    *,
    pot_id: str,
    limit: int = 20,
) -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id=pot_id,
        goal=ContextGraphGoal.AGGREGATE,
        strategy=ContextGraphStrategy.EXACT,
        include=["graph_overview"],
        limit=limit,
    )
