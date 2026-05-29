"""PR diff reader (compat).

Full diffs belong behind source resolvers. The reader is preserved for
legacy callers and stamps ``compat=True`` so consumers can surface the
deprecation hint.
"""

from __future__ import annotations

from adapters.outbound.graphiti.query_helpers import get_pr_diff
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class PrDiffReader:
    FAMILY = "pr_diff"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="PR diff rows (compat — prefer source_policy=snippets via context_resolve).",
            intents=frozenset(),
            requires_scope=frozenset({"pr_number"}),
            cost=ReaderCost(label="medium", estimated_ms=200),
            backend="structural",
            compat=True,
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        rows = get_pr_diff(
            self._structural,
            request.pot_id,
            request.scope.pr_number or 0,
            file_path=request.scope.file_path,
            limit=request.limit,
            repo_name=request.scope.repo_name,
        )
        return ReaderResult(
            family=self.FAMILY,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
            compat=True,
        )


__all__ = ["PrDiffReader"]
