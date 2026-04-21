"""Unified context graph adapter backed by Graphiti + canonical Neo4j reads.

This is intentionally a delegating adapter for the first migration slice: the
application gets one query method now, while existing proven exact/semantic read
implementations continue to execute underneath it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    get_graph_overview,
    get_pr_diff,
    get_pr_review_context,
    get_project_graph,
    search_pot_context,
)
from application.use_cases.resolve_context import resolve_context
from domain.agent_context_port import bundle_to_agent_envelope
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphStrategy,
)
from domain.intelligence_models import (
    ArtifactRef,
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)
from domain.ports.context_graph import ContextGraphPort
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.structural_graph import StructuralGraphPort


class GraphitiContextGraphAdapter(ContextGraphPort):
    """One graph query surface over Graphiti semantic and canonical exact reads."""

    def __init__(
        self,
        *,
        episodic: EpisodicGraphPort,
        structural: StructuralGraphPort,
        resolution_service: Any | None = None,
    ) -> None:
        self._episodic = episodic
        self._structural = structural
        self._resolution_service = resolution_service

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._episodic, "enabled", False))

    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal == ContextGraphGoal.ANSWER:
            return asyncio.run(self.query_async(request))
        return self._query_sync(request)

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal == ContextGraphGoal.ANSWER:
            return await self._answer_async(request)
        return self._query_sync(request)

    def _query_sync(self, request: ContextGraphQuery) -> ContextGraphResult:
        if self._is_semantic_retrieve(request):
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
            return self._result(request, "semantic_search", rows)

        if request.goal == ContextGraphGoal.TIMELINE:
            rows = get_change_history(
                self._structural,
                request.pot_id,
                function_name=request.scope.function_name,
                file_path=request.scope.file_path,
                limit=request.limit,
                repo_name=request.scope.repo_name,
                pr_number=request.scope.pr_number,
                as_of=request.as_of,
            )
            return self._result(request, "change_history", rows)

        include = set(request.include)
        if (
            request.goal == ContextGraphGoal.AGGREGATE
            and "owners" in include
            and request.scope.file_path
        ):
            rows = get_file_owners(
                self._structural,
                request.pot_id,
                request.scope.file_path or "",
                limit=min(request.limit, 50),
                repo_name=request.scope.repo_name,
            )
            return self._result(request, "file_owners", rows)

        if "decisions" in include:
            rows = get_decisions(
                self._structural,
                request.pot_id,
                file_path=request.scope.file_path,
                function_name=request.scope.function_name,
                limit=request.limit,
                repo_name=request.scope.repo_name,
                pr_number=request.scope.pr_number,
            )
            return self._result(request, "decisions", rows)

        if "pr_review_context" in include:
            out = get_pr_review_context(
                self._structural,
                request.pot_id,
                request.scope.pr_number or 0,
                repo_name=request.scope.repo_name,
            )
            return self._result(request, "pr_review_context", out)

        if "pr_diff" in include:
            rows = get_pr_diff(
                self._structural,
                request.pot_id,
                request.scope.pr_number or 0,
                file_path=request.scope.file_path,
                limit=request.limit,
                repo_name=request.scope.repo_name,
            )
            return self._result(request, "pr_diff", rows)

        if request.goal == ContextGraphGoal.NEIGHBORHOOD:
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
            return self._result(request, "project_graph", out)

        if request.goal == ContextGraphGoal.AGGREGATE:
            out = get_graph_overview(
                self._structural,
                self._episodic,
                request.pot_id,
                top_entities_limit=min(request.limit, 100),
            )
            return self._result(request, "graph_overview", out)

        return ContextGraphResult(
            kind=request.goal.value,
            goal=request.goal.value,
            strategy=request.strategy.value,
            error="unsupported_context_graph_query",
            meta={"include": request.include},
        )

    async def _answer_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        if self._resolution_service is None:
            return ContextGraphResult(
                kind="resolve_context",
                goal=request.goal.value,
                strategy=request.strategy.value,
                error="resolution_service_unavailable",
            )
        art = (
            ArtifactRef(
                kind=request.artifact.kind,
                identifier=request.artifact.identifier,
            )
            if request.artifact
            else None
        )
        scope = ContextScope(
            repo_name=request.scope.repo_name,
            branch=request.scope.branch,
            file_path=request.scope.file_path,
            function_name=request.scope.function_name,
            symbol=request.scope.symbol,
            pr_number=request.scope.pr_number,
            services=list(request.scope.services),
            features=list(request.scope.features),
            environment=request.scope.environment,
            ticket_ids=list(request.scope.ticket_ids),
            user=request.scope.user,
            source_refs=list(request.scope.source_refs),
        )
        req = ContextResolutionRequest(
            pot_id=request.pot_id,
            query=request.query or "",
            consumer_hint=request.consumer_hint,
            artifact_ref=art,
            scope=scope,
            intent=request.intent,
            include=request.include,
            exclude=request.exclude,
            mode=_mode_for_strategy(request.strategy),
            source_policy=request.source_policy,
            budget=ContextBudget(
                max_items=request.budget.max_items,
                max_tokens=request.budget.max_tokens,
                timeout_ms=request.budget.timeout_ms,
                freshness=request.budget.freshness,
            ),
            as_of=request.as_of,
            timeout_ms=request.budget.timeout_ms,
        )
        bundle = await resolve_context(self._resolution_service, req)
        return self._result(
            request,
            "resolve_context",
            bundle_to_agent_envelope(bundle),
        )

    @staticmethod
    def _is_semantic_retrieve(request: ContextGraphQuery) -> bool:
        if request.strategy in {
            ContextGraphStrategy.SEMANTIC,
            ContextGraphStrategy.HYBRID,
        }:
            return bool(request.query and request.query.strip())
        return "semantic_search" in set(request.include) and bool(
            request.query and request.query.strip()
        )

    @staticmethod
    def _result(
        request: ContextGraphQuery,
        kind: str,
        result: Any,
    ) -> ContextGraphResult:
        return ContextGraphResult(
            kind=kind,
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=result,
        )


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _mode_for_strategy(strategy: ContextGraphStrategy) -> str:
    if strategy == ContextGraphStrategy.HYBRID:
        return "balanced"
    return "fast"
