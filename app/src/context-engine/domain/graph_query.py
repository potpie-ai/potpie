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
