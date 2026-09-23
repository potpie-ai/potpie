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
from dataclasses import replace
from datetime import datetime
import json
from typing import Any, Mapping, Sequence

from .resource_projection import project_public_metadata


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
                    "completeness": self.metadata.get("readers", {}).get(report.include, {}).get("completeness", "unknown"),
                    "page_status": report.status,
                    "candidate_pool": report.candidate_pool,
                    "candidate_pool_unit": self.metadata.get("readers", {}).get(report.include, {}).get("candidate_pool_unit", "reader_candidates"),
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


DEFAULT_OUTPUT_BUDGET_BYTES = 32_768


def bound_agent_envelope(
    envelope: AgentEnvelope, *, max_bytes: int = DEFAULT_OUTPUT_BUDGET_BYTES
) -> AgentEnvelope:
    """Bound serialized evidence while preserving identity and answer fields."""
    def size(value: AgentEnvelope) -> int:
        return len(json.dumps(value.to_dict(), ensure_ascii=False, default=str).encode("utf-8"))

    envelope = replace(envelope, items=tuple(
        replace(item, payload=project_public_metadata(item.payload))
        for item in envelope.items
    ))
    if size(envelope) <= max_bytes:
        return envelope
    items = list(envelope.items)
    omitted: dict[str, int] = {}
    metadata = dict(envelope.metadata)
    metadata["output_budget_bytes"] = max_bytes

    def current() -> AgentEnvelope:
        metadata["omitted_by_output_budget"] = dict(omitted)
        returned = dict(metadata.get("returned_by_family") or {})
        if returned:
            metadata["returned_by_family"] = {
                family: sum(item.include == family for item in items) for family in returned
            }
        metadata["more_results_available"] = True
        return replace(envelope, items=tuple(items), metadata=dict(metadata))

    while len(items) > 1 and size(current()) > max_bytes:
        removed = items.pop()
        omitted[removed.include] = omitted.get(removed.include, 0) + 1
    if size(current()) <= max_bytes:
        return current()
    if not items:
        return current()

    item = items[0]
    original = dict(item.payload)
    for text_limit, list_limit in ((2_000, 4), (500, 2), (160, 1)):
        bounded, omitted_fields = _bound_evidence_payload(
            original, text_limit=text_limit, list_limit=list_limit,
        )
        items[0] = replace(item, payload=bounded)
        metadata["omitted_fields_by_candidate"] = {
            item.candidate_key: omitted_fields
        } if omitted_fields else {}
        if size(current()) <= max_bytes:
            return current()
    compact = dict(items[0].payload)
    properties = compact.get("properties")
    if isinstance(properties, Mapping):
        essential = {
            "name", "summary", "description", "root_cause", "fix_steps",
            "verification_status", "resolution_status", "source_ref", "revision",
        }
        compact["properties"] = {
            key: value for key, value in properties.items() if key in essential
        }
        metadata.setdefault("omitted_fields_by_candidate", {}).setdefault(
            item.candidate_key, {}
        )["properties"] = len(properties) - len(compact["properties"])
    items[0] = replace(item, payload=compact)
    if size(current()) <= max_bytes:
        return current()
    bounded, omitted_fields = _bound_evidence_payload(
        compact, text_limit=160, list_limit=1, mapping_limit=12,
    )
    items[0] = replace(item, payload=bounded)
    metadata["omitted_fields_by_candidate"] = {item.candidate_key: omitted_fields}
    if size(current()) <= max_bytes:
        return current()
    compact_metadata = {
        key: metadata[key]
        for key in (
            "searched_families", "total_result_budget", "omitted_by_total_budget",
            "omitted_by_output_budget", "more_results_available", "output_budget_bytes",
        )
        if key in metadata
    }
    compact_metadata["omitted_metadata_fields"] = len(metadata) - len(compact_metadata)
    metadata.clear()
    metadata.update(compact_metadata)
    if size(current()) <= max_bytes:
        return current()
    # An unbounded identifier or answer may itself exceed the envelope budget.
    metadata["omitted_candidate_key"] = item.candidate_key
    items.clear()
    omitted[item.include] = omitted.get(item.include, 0) + 1
    return current()


def _bound_evidence_payload(
    payload: Mapping[str, Any], *, text_limit: int, list_limit: int,
    mapping_limit: int = 32,
) -> tuple[dict[str, Any], dict[str, int]]:
    omitted: dict[str, int] = {}
    protected = {"resource_id", "doc", "section", "revision", "source_ref", "claim_key",
                 "subject_key", "object_key", "entity_key", "follow_up_commands"}

    def bound(value: Any, path: str) -> Any:
        if isinstance(value, str):
            if len(value) > text_limit and path.split(".")[-1] not in protected:
                omitted[path] = len(value) - text_limit
                return value[:text_limit]
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > list_limit:
                omitted[path] = len(value) - list_limit
                value = value[:list_limit]
            return [bound(item, f"{path}.{index}") for index, item in enumerate(value)]
        if isinstance(value, Mapping):
            essential = {
                "resource_id", "doc", "section", "revision", "source_ref",
                "source_refs", "claim_key", "subject_key", "object_key",
                "entity_key", "follow_up_commands", "details", "root_cause",
                "fix_steps", "verification_status", "resolution_status",
                "source_status",
            }
            keys = [key for key in value if key in essential]
            keys.extend(key for key in value if key not in essential)
            selected = keys[:mapping_limit]
            if len(keys) > len(selected):
                omitted[path or "properties"] = len(keys) - len(selected)
            return {key: bound(value[key], f"{path}.{key}" if path else str(key))
                    for key in selected}
        return value

    return bound(payload, ""), omitted


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
