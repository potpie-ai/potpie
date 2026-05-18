"""Project-graph reader: bounded neighborhood traversal around a scope."""

from __future__ import annotations

from adapters.outbound.graphiti.query_helpers import get_project_graph
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


class ProjectGraphReader:
    FAMILY = "project_graph"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="Bounded neighborhood traversal anchored on PR / repo / service / feature scope.",
            intents=frozenset({"onboarding", "refactor", "planning"}),
            requires_scope=frozenset(),
            cost=ReaderCost(label="medium", estimated_ms=250),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        out = get_project_graph(
            self._structural,
            request.pot_id,
            pr_number=request.scope.pr_number,
            limit=min(request.limit, 50),
            scope={
                "repo_name": request.scope.repo_name,
                "services": request.scope.services,
                "features": request.scope.features,
                "environment": request.scope.environment,
                "user": request.scope.user,
            },
            include=request.include,
        )
        return ReaderResult(family=self.FAMILY, result=out)


__all__ = ["ProjectGraphReader"]
