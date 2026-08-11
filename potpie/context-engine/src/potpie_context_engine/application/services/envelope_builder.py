"""Build the canonical :class:`AgentEnvelope` from reader responses (P8).

This service is the single envelope-shaper for the read path. The
:class:`ReadOrchestrator` hands in a list of ``(include, ReadResponse)``
pairs and the resolved intent; the builder sorts cross-include by ranker
score, computes per-include coverage, and rolls up the overall confidence
per F5. It is the only envelope shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from potpie_context_engine.application.readers._common import ReadResponse
from potpie_context_core.agent_context_port import (
    includes_for_request,
    normalize_context_intent,
    unsupported_include_values,
)
from potpie_context_core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
    UnsupportedInclude,
    derive_overall_confidence,
)
from potpie_context_core.graph_views import INCLUDE_TO_VIEW


# Cross-include demotion so a doc corpus cannot crowd project memory out of a
# mixed envelope (resources P9). Applied only here — DocsReader itself is
# unchanged, so a pure ``--include docs`` read keeps its internal ranking.
INCLUDE_RANK_WEIGHT: Mapping[str, float] = {
    "prior_bugs": 1.0,
    "decisions": 1.0,
    "coding_preferences": 1.0,
    "features": 1.0,
    "owners": 1.0,
    "infra_topology": 1.0,
    "timeline": 1.0,
    "docs": 0.65,
    # Below ``docs`` on purpose. A document corpus produces far more matchable
    # text than project memory does, so at equal weight a mixed envelope fills
    # with chunk hits and the decisions and prior bugs the agent asked for fall
    # off the end — the same crowding that demoting ``docs`` to 0.65 fixed, one
    # order of magnitude worse because the unit here is a chunk rather than a
    # section. A pure ``--include resources`` read is unaffected: the weight is
    # applied across includes, never inside the reader.
    "resources": 0.6,
    "raw_graph": 0.5,
}


def include_rank_weight(include: str) -> float:
    return float(INCLUDE_RANK_WEIGHT.get(include, 1.0))


@dataclass(slots=True)
class IncludeResult:
    """One reader's contribution to the envelope."""

    include: str
    response: ReadResponse


@dataclass(slots=True)
class EnvelopeBuilder:
    """Stateless service. Inject custom intent/include mappings via constructor."""

    additional_includes: frozenset[str] = frozenset()

    def build(
        self,
        *,
        pot_id: str,
        intent: str,
        results: Iterable[IncludeResult],
        requested_includes: Sequence[str] | None = None,
        extra_unsupported: Sequence[UnsupportedInclude] = (),
        as_of: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentEnvelope:
        intent = normalize_context_intent(intent)
        requested_list = list(requested_includes or [])
        # Resolve against the canonical vocabulary: empty request → the
        # intent's default includes; unknown names → ``unsupported`` (never
        # silently dropped to zero).
        resolved = includes_for_request(intent, requested_list, [])
        unsupported_names = [
            name
            for name in unsupported_include_values(requested_list)
            if name not in self.additional_includes
        ]
        unsupported_set = set(unsupported_names)
        matched = [inc for inc in resolved if inc not in unsupported_set]
        matched_set = set(matched)
        # Unknown names (not in the vocab) + caller-supplied not-implemented
        # entries (in-vocab includes the orchestrator had no reader for).
        unsupported_raw = tuple(
            UnsupportedInclude(name=name, reason="unknown_include")
            for name in unsupported_names
        ) + tuple(extra_unsupported)

        items: list[EvidenceItem] = []
        coverage: list[CoverageReport] = []
        for include_result in results:
            inc = include_result.include
            if matched and inc not in matched_set:
                # Reader produced output but caller didn't ask for it
                # under this intent. Skip silently.
                continue
            resp = include_result.response
            weight = include_rank_weight(inc)
            for ranked in resp.items:
                items.append(
                    EvidenceItem(
                        include=inc,
                        candidate_key=ranked.candidate.candidate_key,
                        score=ranked.score * weight,
                        payload=dict(ranked.candidate.payload),
                        coverage_status=resp.coverage_status,
                        breakdown=dict(ranked.breakdown),
                    )
                )
            coverage.append(
                CoverageReport(
                    include=inc,
                    status=resp.coverage_status,
                    candidate_pool=int(resp.meta.get("candidate_pool", 0))
                    if isinstance(resp.meta.get("candidate_pool"), int)
                    else 0,
                    graph_view=INCLUDE_TO_VIEW.get(inc),
                )
            )

        items.sort(key=lambda i: i.score, reverse=True)

        return AgentEnvelope(
            pot_id=pot_id,
            intent=intent,
            items=tuple(items),
            coverage=tuple(coverage),
            unsupported_includes=tuple(unsupported_raw),
            overall_confidence=derive_overall_confidence(coverage=coverage),
            as_of=as_of,
            metadata=dict(metadata or {}),
        )


def envelope_to_dict(envelope: AgentEnvelope) -> dict[str, object]:
    """Serialise the canonical envelope to a JSON-shaped dict.

    Used by the HTTP boundary to send the envelope on the wire; ``intent``
    and ``include`` are already canonical strings.
    """
    return envelope.to_dict()


__all__ = [
    "INCLUDE_RANK_WEIGHT",
    "EnvelopeBuilder",
    "IncludeResult",
    "envelope_to_dict",
    "include_rank_weight",
    "UnsupportedInclude",
]
