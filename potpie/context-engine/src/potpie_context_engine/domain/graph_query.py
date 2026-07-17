"""Unified graph query request model."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContextGraphGoal(StrEnum):
    # Structural read hint only — never selects a different read path. The
    # former ANSWER/INVESTIGATE members (server-side synthesis / agentic loop)
    # were removed when the engine collapsed onto one evidence-envelope read
    # contract; the agent synthesises answers from the returned evidence.
    RETRIEVE = "retrieve"
    TIMELINE = "timeline"


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
    mutation_ids: list[str] = Field(default_factory=list)
    """Filter results to facts written by these apply-plan mutation UUIDs."""
    as_of: datetime | None = None
    # Optional temporal window for timeline queries. ``since`` and ``until``
    # compose with ``as_of`` (which remains a point-in-time snapshot pointer
    # used by bi-temporal reads). When only ``window`` is provided it is
    # interpreted relative to ``until`` or ``as_of`` or ``now``.
    since: datetime | None = None
    until: datetime | None = None
    window: str | None = None  # "7d", "24h", "30d"
    verbs: list[str] = Field(default_factory=list)
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


# Read-tool presets. Every preset emits an include the ReadOrchestrator backs
# (or [] for the generic, intent-routed search). Keeping them in one place means
# the agent read-tool surface and the orchestrator cannot drift: a preset naming
# an unbacked include would surface as ``unsupported_include`` instead of data.
# The tool-name → include mapping lives in
# ``potpie_context_engine.adapters.outbound.reconciliation.context_graph_tools.READ_TOOL_INCLUDE``.


def preset_context_search(
    *,
    pot_id: str,
    query: str,
    intent: str | None = None,
    include: list[str] | None = None,
    repo_name: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    pr_number: int | None = None,
    node_labels: list[str] | None = None,
    include_invalidated: bool = False,
    as_of: datetime | None = None,
    limit: int = 12,
) -> ContextGraphQuery:
    """Generic lookup: the orchestrator routes by ``intent`` (or an explicit
    ``include`` list) into its reader families."""
    return ContextGraphQuery(
        pot_id=pot_id,
        query=query,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.AUTO,
        intent=intent,
        include=list(include or []),
        scope=ContextGraphScope(
            repo_name=repo_name,
            file_path=file_path,
            function_name=function_name,
            pr_number=pr_number,
        ),
        node_labels=list(node_labels or []),
        include_invalidated=include_invalidated,
        as_of=as_of,
        limit=limit,
    )


def preset_reader_lookup(
    *,
    pot_id: str,
    include: str,
    query: str | None = None,
    repo_name: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    pr_number: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    include_invalidated: bool = False,
    as_of: datetime | None = None,
    limit: int = 12,
) -> ContextGraphQuery:
    """Target a single P9 reader by its include family (one of
    coding_preferences / infra_topology / timeline / prior_bugs)."""
    return ContextGraphQuery(
        pot_id=pot_id,
        query=query,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.AUTO,
        include=[include],
        scope=ContextGraphScope(
            repo_name=repo_name,
            file_path=file_path,
            function_name=function_name,
            pr_number=pr_number,
        ),
        since=since,
        until=until,
        include_invalidated=include_invalidated,
        as_of=as_of,
        limit=limit,
    )
