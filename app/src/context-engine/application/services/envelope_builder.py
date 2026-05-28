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

from application.readers._common import ReadResponse
from domain.agent_context_port import (
    includes_for_request,
    normalize_context_intent,
    unsupported_include_values,
)
from domain.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
    UnsupportedInclude,
    derive_overall_confidence,
)


@dataclass(slots=True)
class IncludeResult:
    """One reader's contribution to the envelope."""

    include: str
    response: ReadResponse


@dataclass(slots=True)
class EnvelopeBuilder:
    """Stateless service. Inject custom intent/include mappings via constructor."""

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
        unsupported_names = unsupported_include_values(requested_list)
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
            for ranked in resp.items:
                items.append(
                    EvidenceItem(
                        include=inc,
                        candidate_key=ranked.candidate.candidate_key,
                        score=ranked.score,
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

    Used by the HTTP/MCP boundary to send the envelope on the wire; ``intent``
    and ``include`` are already canonical strings.
    """
    return {
        "pot_id": envelope.pot_id,
        "intent": envelope.intent,
        "items": [
            {
                "include": item.include,
                "candidate_key": item.candidate_key,
                "score": item.score,
                "payload": dict(item.payload),
                "coverage_status": item.coverage_status,
                "breakdown": dict(item.breakdown),
            }
            for item in envelope.items
        ],
        "coverage": [
            {
                "include": c.include,
                "status": c.status,
                "candidate_pool": c.candidate_pool,
            }
            for c in envelope.coverage
        ],
        "unsupported_includes": [
            {"name": u.name, "reason": u.reason}
            for u in envelope.unsupported_includes
        ],
        "overall_confidence": envelope.overall_confidence,
        "as_of": envelope.as_of.isoformat() if envelope.as_of else None,
        "metadata": dict(envelope.metadata),
    }


__all__ = [
    "EnvelopeBuilder",
    "IncludeResult",
    "envelope_to_dict",
    "UnsupportedInclude",
]
