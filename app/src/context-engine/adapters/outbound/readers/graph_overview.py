"""Graph-overview reader: schema coverage and ontology drift signal."""

from __future__ import annotations

from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.graphiti.query_helpers import get_graph_overview
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class GraphOverviewReader:
    FAMILY = "graph_overview"

    def __init__(
        self,
        *,
        episodic: EpisodicGraphPort,
        structural: StructuralReadPort,
    ) -> None:
        self._episodic = episodic
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="Schema-health overview: label / edge coverage and drift signal.",
            intents=frozenset({"onboarding", "operations"}),
            requires_scope=frozenset(),
            cost=ReaderCost(label="medium", estimated_ms=300),
            backend="hybrid",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        out = get_graph_overview(
            self._structural,
            self._episodic,
            request.pot_id,
            top_entities_limit=min(request.limit, 100),
        )
        return ReaderResult(family=self.FAMILY, result=out)


__all__ = ["GraphOverviewReader"]
