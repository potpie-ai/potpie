"""Semantic-search reader: episodic vector search over the pot."""

from __future__ import annotations

from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.graphiti.query_helpers import search_pot_context
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class SemanticSearchReader:
    """Vector search across the pot's existing facts."""

    FAMILY = "semantic_search"

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
            description="Vector search across the pot's existing facts.",
            intents=frozenset({"unknown", "feature", "debugging"}),
            requires_scope=frozenset({"query"}),
            cost=ReaderCost(label="medium", estimated_ms=400),
            backend="graphiti",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = search_pot_context(
            self._episodic,
            request.pot_id,
            request.query or "",
            limit=request.limit,
            node_labels=request.node_labels or None,
            repo_name=request.scope.repo_name,
            source_description=_first(request.source_descriptions),
            include_invalidated=request.include_invalidated,
            as_of=request.as_of,
            episode_uuid=_first(request.episode_uuids),
            structural=self._structural,
        )
        return ReaderResult(
            family=self.FAMILY,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


__all__ = ["SemanticSearchReader"]
