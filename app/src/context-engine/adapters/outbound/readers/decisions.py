"""Decisions reader: durable decisions captured against this pot."""

from __future__ import annotations

from typing import Any

from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.graphiti.query_helpers import get_decisions
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class DecisionsReader:
    FAMILY = "decisions"

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
            description="Durable decisions captured against this pot.",
            intents=frozenset(
                {
                    "feature",
                    "review",
                    "planning",
                    "refactor",
                    "docs",
                    "security",
                    "test",
                    "unknown",
                }
            ),
            requires_scope=frozenset(),
            cost=ReaderCost(label="cheap", estimated_ms=120),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = get_decisions(
            self._structural,
            request.pot_id,
            file_path=request.scope.file_path,
            function_name=request.scope.function_name,
            limit=request.limit,
            repo_name=request.scope.repo_name,
            pr_number=request.scope.pr_number,
            episodic=self._episodic,
            query=request.query,
        )
        return ReaderResult(
            family=self.FAMILY,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
            fallback_reason=_semantic_fallback_reason(rows),
        )


def _semantic_fallback_reason(rows: Any) -> str | None:
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        if rows[0].get("source_method") == "semantic_fallback":
            return "semantic_fallback"
    return None


__all__ = ["DecisionsReader"]
