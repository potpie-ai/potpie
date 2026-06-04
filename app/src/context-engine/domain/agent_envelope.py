"""Canonical agent envelope (rebuild plan P8).

The one read-result shape, returned by both ``context_resolve`` and
``context_search`` across every surface (CLI / MCP / managed HTTP). It carries
ranked evidence items + coverage; there is no server-side answer synthesis —
the agent reasons over the evidence. This module defines that shape; the
application layer's :class:`EnvelopeBuilder` produces it from ranked reader
responses.

The intent / include *vocabulary* lives in one place — ``domain.agent_context_port``
(``CONTEXT_INTENTS`` + the reader-backed include tiers). This module used to
carry a second, smaller copy (``AgentIntent``/``AgentInclude``/``INTENT_INCLUDES``);
that duplicate was removed so there is a single source of truth. ``intent`` and
``include`` are plain canonical strings here, validated by ``agent_context_port``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One ranked piece of evidence the envelope returns."""

    include: str
    candidate_key: str
    score: float
    payload: Mapping[str, Any]
    coverage_status: str
    breakdown: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Per-include coverage tracking, feeding the envelope's overall confidence."""

    include: str
    status: str  # 'complete' | 'partial' | 'sparse' | 'empty'
    candidate_pool: int = 0


@dataclass(frozen=True, slots=True)
class UnsupportedInclude:
    """An include the caller asked for that the orchestrator could not route."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentEnvelope:
    """The single canonical envelope shape (P8). ``intent`` is a canonical
    intent string from ``agent_context_port.CONTEXT_INTENTS``."""

    pot_id: str
    intent: str
    items: tuple[EvidenceItem, ...]
    coverage: tuple[CoverageReport, ...]
    unsupported_includes: tuple[UnsupportedInclude, ...] = ()
    overall_confidence: str = "unknown"  # 'high' | 'medium' | 'low' | 'unknown'
    as_of: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    "CoverageReport",
    "EvidenceItem",
    "UnsupportedInclude",
    "derive_overall_confidence",
]
