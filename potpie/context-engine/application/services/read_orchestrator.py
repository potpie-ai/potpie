"""The single read trunk (rebuild plan P8 + P9).

One orchestrator is the entire read path. It does exactly four things:

1. expand ``intent`` → include families (``includes_for_request``),
2. route each include to its P9 reader (the ``_ROUTING`` table),
3. run each reader over the canonical claim store (``ClaimQueryPort``),
4. assemble one ranked :class:`AgentEnvelope` (P8 shape).

Includes that resolve to no reader are surfaced as
``unsupported_include`` (reason ``not_implemented``) rather than silently
returning nothing — the plan's anti-phantom-vocabulary rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from application.readers._common import ReadRequest
from application.readers.coding_preferences import CodingPreferencesReader
from application.readers.decisions import DecisionsReader
from application.readers.docs import DocsReader
from application.readers.features import FeaturesReader
from application.readers.infra_topology import InfraTopologyReader
from application.readers.owners import OwnersReader
from application.readers.prior_bugs import PriorBugsReader
from application.readers.raw_graph import RawGraphReader
from application.readers.timeline_reader import TimelineReader
from application.services.envelope_builder import (
    EnvelopeBuilder,
    IncludeResult,
)
from domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    includes_for_request,
    normalize_context_intent,
)
from domain.agent_envelope import AgentEnvelope, UnsupportedInclude
from domain.ports.claim_query import ClaimQueryPort
from domain.ranking import RankingService


# Protocol-free reader alias: every P9 reader exposes ``read(ReadRequest)``.
_ReaderT = Any


@dataclass(slots=True)
class ReadOrchestrator:
    """Build P9 readers over a claim store and resolve a task into an envelope."""

    claim_query: ClaimQueryPort
    ranker: RankingService = field(default_factory=RankingService)
    builder: EnvelopeBuilder = field(default_factory=EnvelopeBuilder)
    _routing: dict[str, _ReaderT] = field(init=False)

    def __post_init__(self) -> None:
        cq, rk = self.claim_query, self.ranker
        # include family → reader. One reader can back multiple includes.
        self._routing = {
            "coding_preferences": CodingPreferencesReader(claim_query=cq, ranker=rk),
            "features": FeaturesReader(claim_query=cq, ranker=rk),
            "infra_topology": InfraTopologyReader(claim_query=cq, ranker=rk),
            "timeline": TimelineReader(claim_query=cq, ranker=rk),
            "prior_bugs": PriorBugsReader(claim_query=cq, ranker=rk),
            "decisions": DecisionsReader(claim_query=cq, ranker=rk),
            "owners": OwnersReader(claim_query=cq, ranker=rk),
            "docs": DocsReader(claim_query=cq, ranker=rk),
            # Visualization read: the whole canonical partition (all RELATES_TO,
            # incl. generic RELATED_TO) for the graph explorer — not a UC slice.
            "raw_graph": RawGraphReader(claim_query=cq, ranker=rk),
        }

    @property
    def backed_includes(self) -> frozenset[str]:
        return frozenset(self._routing)

    def resolve(
        self,
        *,
        pot_id: str,
        intent: str | None = None,
        query: str | None = None,
        scope: Mapping[str, Any] | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        as_of: datetime | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_items: int = 12,
        freshness_preference: str = "balanced",
        include_invalidated: bool = False,
        source_refs: tuple[str, ...] = (),
        query_threshold: float = 0.70,
        depth: int | None = None,
        direction: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentEnvelope:
        intent = normalize_context_intent(intent)
        resolved = includes_for_request(intent, include or [], exclude or [])
        req = ReadRequest(
            pot_id=pot_id,
            scope=dict(scope or {}),
            query=query,
            intent=intent,
            as_of=as_of,
            since=since,
            until=until,
            max_items=max_items,
            freshness_preference=freshness_preference,
            include_invalidated=include_invalidated,
            source_refs=tuple(source_refs),
            query_threshold=query_threshold,
            depth=depth,
            direction=direction,
        )
        results: list[IncludeResult] = []
        extra_unsupported: list[UnsupportedInclude] = []
        for inc in resolved:
            reader = self._routing.get(inc)
            if reader is not None:
                results.append(IncludeResult(include=inc, response=reader.read(req)))
            elif inc in CONTEXT_INCLUDE_VALUES:
                # In the vocab but no reader yet → honest not-implemented.
                extra_unsupported.append(
                    UnsupportedInclude(name=inc, reason="not_implemented")
                )
            # else: not in the vocab at all → the EnvelopeBuilder flags it
            # ``unknown_include`` from ``requested_includes``.

        return self.builder.build(
            pot_id=pot_id,
            intent=intent,
            results=results,
            requested_includes=include or None,
            extra_unsupported=extra_unsupported,
            as_of=as_of,
            metadata=metadata,
        )


__all__ = ["ReadOrchestrator"]
