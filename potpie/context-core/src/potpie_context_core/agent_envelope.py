"""Canonical agent envelope (rebuild plan P8).

The one read-result shape, returned by both ``context_resolve`` and
``context_search`` across every surface (CLI / managed HTTP). It carries
ranked evidence items + coverage; there is no server-side answer synthesis —
the agent reasons over the evidence. This module defines that shape; the
application layer's :class:`EnvelopeBuilder` produces it from ranked reader
responses.

The intent / include *vocabulary* lives in one place — ``potpie_context_core.agent_context_port``
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
    graph_view: str | None = None
    """Canonical ``<subgraph>.<view>`` serving this include family — the
    forward pointer that teaches V1 callers the workbench vocabulary."""
    best_relevance: float | None = None
    """Best *absolute, calibrated* relevance this family found, or ``None``.

    ``status`` answers "did the family run and fill the page"; it cannot answer
    "is any of this an answer", because a k-NN fills the page whether or not the
    corpus holds one. This carries the second question, and only a family that
    can measure it on a comparable scale sets it — an uncalibrated index (the
    bundled hashing embedder) leaves it ``None`` rather than reporting a number
    whose magnitude means nothing.
    """


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the canonical envelope to a JSON-shaped dict."""
        return {
            "pot_id": self.pot_id,
            "intent": self.intent,
            "items": [
                {
                    "include": item.include,
                    "candidate_key": item.candidate_key,
                    "score": item.score,
                    "payload": dict(item.payload),
                    "coverage_status": item.coverage_status,
                    "breakdown": dict(item.breakdown),
                }
                for item in self.items
            ],
            "coverage": [
                {
                    "include": report.include,
                    "status": report.status,
                    "candidate_pool": report.candidate_pool,
                    "graph_view": report.graph_view,
                    "best_relevance": report.best_relevance,
                }
                for report in self.coverage
            ],
            "unsupported_includes": [
                {"name": unsupported.name, "reason": unsupported.reason}
                for unsupported in self.unsupported_includes
            ],
            "overall_confidence": self.overall_confidence,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "metadata": dict(self.metadata),
        }


#: Best calibrated relevance at or above which the evidence is called ``high``.
#: Measured on the 202-question retrieval benchmark (190 answerable + 12
#: labelled unanswerable, MiniLM-L6 over a 316-chunk corpus): 50.0% of
#: answerable queries clear this, against 1 of 12 unanswerable.
RELEVANCE_CONFIDENCE_HIGH = 0.50

#: Below this the evidence is called ``low``. Same measurement: it captures
#: 58.3% of unanswerable queries while mislabelling only 7.4% of answerable
#: ones, which is why the band exists at all.
RELEVANCE_CONFIDENCE_MEDIUM = 0.35

_STATUS_RANKS = {"complete": 4, "partial": 3, "sparse": 2, "empty": 1, "unknown": 0}
_RANK_TO_CONFIDENCE = {4: "high", 3: "medium", 2: "low", 1: "low", 0: "unknown"}
_CONFIDENCE_RANKS = {"high": 4, "medium": 3, "low": 2, "unknown": 0}


def relevance_confidence(best_relevance: float | None) -> str:
    """Band an absolute, calibrated relevance. ``None`` is ``unknown``."""
    if best_relevance is None:
        return "unknown"
    if best_relevance >= RELEVANCE_CONFIDENCE_HIGH:
        return "high"
    if best_relevance >= RELEVANCE_CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def derive_overall_confidence(*, coverage: Sequence[CoverageReport]) -> str:
    """Map per-include coverage into the envelope's overall_confidence (F5).

    Two independent questions, and the answer is the worse of them.

    ``status`` reports whether a family ran and filled its page. On its own it
    reported ``high`` for *every* query a working index served — including ones
    the corpus demonstrably could not answer — because a k-NN returns k rows
    whether or not any is an answer, and a full page of them is ``complete``.
    Confidence built from that alone says "I searched successfully", which an
    agent reasonably reads as "this is the answer".

    So a family that can measure *how good* its best hit is contributes that
    too, through :attr:`CoverageReport.best_relevance`, and the envelope takes
    the lower band. A family that cannot measure it on a comparable scale
    reports ``None`` and is judged on coverage exactly as before — which keeps
    an uncalibrated index from being downgraded by a number that does not mean
    what the thresholds assume.

    Deliberately *not* suppression: on the benchmark, rejecting results below
    the ``low`` band would have discarded a real answer for 7.4% of answerable
    queries to silence 58.3% of unanswerable ones, and the cut-off would rest
    on 12 labelled negatives. Reporting the band costs no recall and hands the
    caller the signal to threshold on.
    """
    if not coverage:
        return "unknown"
    worst_status = min(_STATUS_RANKS.get(c.status, 0) for c in coverage)
    confidence = _RANK_TO_CONFIDENCE[worst_status]

    measured = [c.best_relevance for c in coverage if c.best_relevance is not None]
    if not measured:
        return confidence
    by_relevance = relevance_confidence(max(measured))
    if _CONFIDENCE_RANKS[by_relevance] < _CONFIDENCE_RANKS[confidence]:
        return by_relevance
    return confidence


__all__ = [
    "RELEVANCE_CONFIDENCE_HIGH",
    "RELEVANCE_CONFIDENCE_MEDIUM",
    "AgentEnvelope",
    "CoverageReport",
    "EvidenceItem",
    "UnsupportedInclude",
    "derive_overall_confidence",
    "relevance_confidence",
]
