"""PR review-context reader: PR title + review-thread bundle."""

from __future__ import annotations

from adapters.outbound.graphiti.query_helpers import get_pr_review_context
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class PrReviewContextReader:
    FAMILY = "pr_review_context"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="PR title, summary, and grouped review-thread bundle.",
            intents=frozenset({"review"}),
            requires_scope=frozenset({"pr_number"}),
            cost=ReaderCost(label="cheap", estimated_ms=90),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        out = get_pr_review_context(
            self._structural,
            request.pot_id,
            request.scope.pr_number or 0,
            repo_name=request.scope.repo_name,
        )
        return ReaderResult(family=self.FAMILY, result=out)


__all__ = ["PrReviewContextReader"]
