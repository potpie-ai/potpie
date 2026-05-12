"""Change-history reader: temporal PR/commit activity around a code anchor."""

from __future__ import annotations

from typing import Any

from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.graphiti.query_helpers import get_change_history
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class ChangeHistoryReader:
    FAMILY = "change_history"

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
            description="Recent change history for a file, function, or PR in this pot.",
            intents=frozenset({"feature", "debugging", "review", "refactor", "test"}),
            requires_scope=frozenset(),  # falls back via semantic seeds when empty
            cost=ReaderCost(label="cheap", estimated_ms=120),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = get_change_history(
            self._structural,
            request.pot_id,
            function_name=request.scope.function_name,
            file_path=request.scope.file_path,
            limit=request.limit,
            repo_name=request.scope.repo_name,
            pr_number=request.scope.pr_number,
            as_of=request.as_of,
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


__all__ = ["ChangeHistoryReader"]
