"""Canonical agent envelope (rebuild plan P8).

The plan calls for one envelope shape across ``goal=answer|retrieve``
and across MCP ``context_resolve`` / ``context_search``. This module
defines the canonical shape; the application layer's envelope-builder
service produces it from ranked reader responses. The intent / include
catalog is enumerated here so the agent-contract generator can read
the canonical truth from code, not docs.

Until P8's full migration completes, the legacy envelopes coexist with
this canonical one. Callers migrate one reader at a time; the
``EnvelopeBuilder`` is the single source of truth for the new shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence


class AgentIntent(str, Enum):
    """The ~5 intents per the plan (P8 decisions)."""

    FEATURE = "feature"
    DEBUGGING = "debugging"
    OPERATIONS = "operations"
    ONBOARDING = "onboarding"
    REVIEW = "review"


class AgentInclude(str, Enum):
    """The ~10 task-shaped includes the agent vocabulary collapses to."""

    CODING_PREFERENCES = "coding_preferences"
    INFRA_TOPOLOGY = "infra_topology"
    RECENT_CHANGES = "recent_changes"
    PRIOR_BUGS = "prior_bugs"
    PRIOR_FIXES = "prior_fixes"
    DECISIONS = "decisions"
    OWNERS = "owners"
    DOCS = "docs"
    SOURCE_STATUS = "source_status"
    SEMANTIC_SEARCH = "semantic_search"


# Default intent → include routing. Readers/orchestrators consult this
# instead of hard-coding the mapping per intent.
INTENT_INCLUDES: Mapping[AgentIntent, tuple[AgentInclude, ...]] = {
    AgentIntent.FEATURE: (
        AgentInclude.CODING_PREFERENCES,
        AgentInclude.DECISIONS,
        AgentInclude.RECENT_CHANGES,
        AgentInclude.OWNERS,
    ),
    AgentIntent.DEBUGGING: (
        AgentInclude.PRIOR_BUGS,
        AgentInclude.PRIOR_FIXES,
        AgentInclude.RECENT_CHANGES,
        AgentInclude.INFRA_TOPOLOGY,
    ),
    AgentIntent.OPERATIONS: (
        AgentInclude.INFRA_TOPOLOGY,
        AgentInclude.OWNERS,
        AgentInclude.DECISIONS,
        AgentInclude.SOURCE_STATUS,
    ),
    AgentIntent.ONBOARDING: (
        AgentInclude.INFRA_TOPOLOGY,
        AgentInclude.DECISIONS,
        AgentInclude.OWNERS,
        AgentInclude.CODING_PREFERENCES,
        AgentInclude.DOCS,
    ),
    AgentIntent.REVIEW: (
        AgentInclude.CODING_PREFERENCES,
        AgentInclude.DECISIONS,
        AgentInclude.PRIOR_BUGS,
        AgentInclude.RECENT_CHANGES,
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One ranked piece of evidence the envelope returns."""

    include: AgentInclude
    candidate_key: str
    score: float
    payload: Mapping[str, Any]
    coverage_status: str
    breakdown: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Per-include coverage tracking, feeding the envelope's overall confidence."""

    include: AgentInclude
    status: str  # 'complete' | 'partial' | 'sparse' | 'empty'
    candidate_pool: int = 0


@dataclass(frozen=True, slots=True)
class UnsupportedInclude:
    """An include the caller asked for that the orchestrator could not route."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentEnvelope:
    """The single canonical envelope shape (P8)."""

    pot_id: str
    intent: AgentIntent
    items: tuple[EvidenceItem, ...]
    coverage: tuple[CoverageReport, ...]
    unsupported_includes: tuple[UnsupportedInclude, ...] = ()
    overall_confidence: str = "unknown"  # 'high' | 'medium' | 'low' | 'unknown'
    as_of: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def resolve_includes(
    *,
    intent: AgentIntent,
    requested: Sequence[str] | None,
) -> tuple[list[AgentInclude], list[UnsupportedInclude]]:
    """Map the caller's requested includes onto the canonical catalog.

    Returns ``(matched, unsupported)``. Unknown include strings become
    :class:`UnsupportedInclude` entries (per plan exit criterion: an
    unknown include produces ``unsupported_include``, never silent zero).
    Empty/None requested falls back to the intent's defaults.
    """
    matched: list[AgentInclude] = []
    unsupported: list[UnsupportedInclude] = []
    if not requested:
        return list(INTENT_INCLUDES.get(intent, ())), []
    valid_names = {member.value: member for member in AgentInclude}
    for raw in requested:
        if not isinstance(raw, str):
            unsupported.append(
                UnsupportedInclude(name=str(raw), reason="not-a-string")
            )
            continue
        canon = valid_names.get(raw)
        if canon is None:
            unsupported.append(
                UnsupportedInclude(name=raw, reason="unknown_include")
            )
            continue
        if canon not in matched:
            matched.append(canon)
    return matched, unsupported


def derive_overall_confidence(
    *, coverage: Sequence[CoverageReport]
) -> str:
    """Map per-include coverage into the envelope's overall_confidence (F5)."""
    if not coverage:
        return "unknown"
    ranks = {"complete": 4, "partial": 3, "sparse": 2, "empty": 1, "unknown": 0}
    worst = min(ranks.get(c.status, 0) for c in coverage)
    return {4: "high", 3: "medium", 2: "low", 1: "low", 0: "unknown"}[worst]


__all__ = [
    "AgentEnvelope",
    "AgentInclude",
    "AgentIntent",
    "CoverageReport",
    "EvidenceItem",
    "INTENT_INCLUDES",
    "UnsupportedInclude",
    "derive_overall_confidence",
    "resolve_includes",
]
