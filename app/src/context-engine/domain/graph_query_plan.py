"""Execution plan model for ``ContextGraphQuery``.

The planner translates a :class:`ContextGraphQuery` (a declarative read
request) into an :class:`ExecutionPlan`: one or more :class:`QueryLeg`
items that each contribute an evidence family (semantic search, change
history, owners, decisions, project map, graph overview, answer, ...) to
the final result envelope.

This keeps the adapter free of if/else family dispatch: the planner
compiles the request, and an executor runs legs with a bounded budget,
deterministic merge policy, and consistent provenance/fallback metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from domain.graph_query import ContextGraphBudget


class LegStrategy(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    TRAVERSAL = "traversal"
    HYBRID = "hybrid"
    ANSWER = "answer"


@dataclass(frozen=True)
class QueryLeg:
    """One evidence family to execute for a query."""

    name: str
    family: str
    strategy: LegStrategy
    limit: int
    as_of: datetime | None = None
    # Informational: scope fields the executor will read. Used by the
    # planner to skip legs that require missing scope (e.g. ``owners``
    # without ``file_path``) so callers get a clear fallback instead of
    # a misleading empty result.
    requires_scope: frozenset[str] = frozenset()
    compat: bool = False


class MergePolicy(StrEnum):
    # Single-family call: result envelope keeps the family name as ``kind``
    # and ``result`` is the raw payload (back-compat for legacy callers).
    SINGLE = "single"
    # Multi-family bulk call: ``kind="multi"`` and ``result`` is a map of
    # family -> payload, with per-leg metadata/fallbacks in ``meta``.
    MULTI = "multi"


@dataclass(frozen=True)
class ExecutionPlan:
    pot_id: str
    legs: tuple[QueryLeg, ...]
    merge_policy: MergePolicy
    budget: ContextGraphBudget
    # Families the planner wanted to run but couldn't (e.g. missing scope,
    # unsupported include token). Surfaced in the response under
    # ``meta.fallbacks`` so agents can see what was dropped and why.
    planner_fallbacks: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass
class LegOutcome:
    """Result of executing one :class:`QueryLeg`."""

    name: str
    family: str
    strategy: str
    result: Any = None
    error: str | None = None
    count: int | None = None
    fallback_reason: str | None = None
    source_refs: tuple[str, ...] = ()
    compat: bool = False


__all__ = [
    "ExecutionPlan",
    "LegOutcome",
    "LegStrategy",
    "MergePolicy",
    "QueryLeg",
]
