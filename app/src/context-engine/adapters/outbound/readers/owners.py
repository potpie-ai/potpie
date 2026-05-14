"""Owners reader: inferred reviewers / owners for a file."""

from __future__ import annotations

from adapters.outbound.graphiti.query_helpers import get_file_owners
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class OwnersReader:
    FAMILY = "owners"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="Inferred owners / reviewers for a file in this pot.",
            intents=frozenset({"review", "refactor", "operations"}),
            requires_scope=frozenset({"file_path"}),
            cost=ReaderCost(label="cheap", estimated_ms=80),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = get_file_owners(
            self._structural,
            request.pot_id,
            request.scope.file_path or "",
            limit=min(request.limit, 50),
            repo_name=request.scope.repo_name,
        )
        return ReaderResult(
            family=self.FAMILY,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )


__all__ = ["OwnersReader"]
