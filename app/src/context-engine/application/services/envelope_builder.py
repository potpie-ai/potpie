"""Build the canonical :class:`AgentEnvelope` from reader responses (P8).

This service is the single envelope-shaper for the new canonical path.
Callers (HTTP resolve, MCP context_search, CLI) hand in a list of
``(include, ReadResponse)`` pairs and the resolved intent; the builder
sorts cross-include by ranker score, computes per-include coverage,
and rolls up the overall confidence per F5.

The legacy `bundle_to_agent_envelope` / `IntelligenceBundle` shapes
remain in place during the migration window. This builder is the
forward path; new code calls it, old code is migrated incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from application.readers._common import ReadResponse
from domain.agent_envelope import (
    AgentEnvelope,
    AgentInclude,
    AgentIntent,
    CoverageReport,
    EvidenceItem,
    UnsupportedInclude,
    derive_overall_confidence,
    resolve_includes,
)


@dataclass(slots=True)
class IncludeResult:
    """One reader's contribution to the envelope."""

    include: AgentInclude
    response: ReadResponse


@dataclass(slots=True)
class EnvelopeBuilder:
    """Stateless service. Inject custom intent/include mappings via constructor."""

    def build(
        self,
        *,
        pot_id: str,
        intent: AgentIntent,
        results: Iterable[IncludeResult],
        requested_includes: Sequence[str] | None = None,
        as_of: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentEnvelope:
        matched, unsupported_raw = resolve_includes(
            intent=intent, requested=requested_includes
        )
        matched_set = set(matched)

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

    Used by the HTTP/MCP boundary to send the envelope on the wire; the
    intent + include enums are serialised as their string values so the
    receiving side doesn't need to import the enum classes.
    """
    return {
        "pot_id": envelope.pot_id,
        "intent": envelope.intent.value,
        "items": [
            {
                "include": item.include.value,
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
                "include": c.include.value,
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
