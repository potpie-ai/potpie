"""Unified context graph adapter backed by Graphiti + canonical Neo4j reads.

Reads go through :class:`GraphQueryPlanner` → per-leg executors → a
single :class:`ContextGraphResult` envelope. Writes remain bounded:
``apply_plan`` for validated reconciliation plan slices and
``write_raw_episode`` for direct episode writes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

from application.services.graph_query_planner import GraphQueryPlanner
from application.use_cases.apply_reconciliation_plan import apply_reconciliation_plan
from application.use_cases.ingest_episode import ingest_episode as ingest_episode_uc
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
from domain.graph_query_plan import (
    ExecutionPlan,
    LegOutcome,
    MergePolicy,
    QueryLeg,
)
from domain.graph_mutations import ProvenanceContext
from domain.intelligence_models import (
    ArtifactRef,
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)
from domain.ports.context_graph import ContextGraphPort
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import ReconciliationPlan, ReconciliationResult


class GraphitiContextGraphAdapter(ContextGraphPort):
    """One graph query surface over Graphiti semantic and canonical exact reads."""

    def __init__(
        self,
        *,
        episodic: EpisodicGraphPort,
        structural: StructuralGraphPort,
        resolution_service: Any | None = None,
        mutation_applier: GraphMutationApplierPort | None = None,
        planner: GraphQueryPlanner | None = None,
    ) -> None:
        self._episodic = episodic
        self._structural = structural
        self._resolution_service = resolution_service
        self._mutation_applier = mutation_applier
        self._planner = planner or GraphQueryPlanner()
        self._executors: dict[str, Callable[[QueryLeg, ContextGraphQuery], LegOutcome]] = {
            "semantic_search": self._exec_semantic_search,
            "change_history": self._exec_change_history,
            "owners": self._exec_owners,
            "decisions": self._exec_decisions,
            "pr_review_context": self._exec_pr_review_context,
            "pr_diff": self._exec_pr_diff,
            "project_graph": self._exec_project_graph,
            "graph_overview": self._exec_graph_overview,
        }

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._episodic, "enabled", False))

    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal == ContextGraphGoal.ANSWER:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.query_async(request))
            raise RuntimeError(
                "GraphitiContextGraphAdapter.query() cannot execute answer "
                "queries while an event loop is already running; use "
                "query_async()."
            )
        return self._execute(self._planner.plan(request), request)

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        plan = self._planner.plan(request)
        if plan.legs and plan.legs[0].name == "answer":
            return await self._answer_async(plan.legs[0], request, plan)
        return self._execute(plan, request)

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        return apply_reconciliation_plan(
            self._episodic,
            self._structural,
            plan,
            expected_pot_id=expected_pot_id,
            mutation_applier=self._mutation_applier,
            provenance_context=provenance_context,
        )

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> dict[str, Any]:
        return ingest_episode_uc(
            self._episodic,
            pot_id,
            name,
            episode_body,
            source_description,
            reference_time,
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"pot_id": pot_id, "ok": False}

        ep = self._episodic.reset_pot(pot_id)
        out["episodic"] = ep
        if not ep.get("ok"):
            out["error"] = ep.get("error", "episodic_reset_failed")
            return out

        st = self._structural.reset_pot(pot_id)
        out["structural"] = st
        if not st.get("ok"):
            out["error"] = st.get("error", "structural_reset_failed")
            return out

        out["ok"] = True
        return out

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute(
        self,
        plan: ExecutionPlan,
        request: ContextGraphQuery,
    ) -> ContextGraphResult:
        if not plan.legs:
            return ContextGraphResult(
                kind=request.goal.value,
                goal=request.goal.value,
                strategy=request.strategy.value,
                error="unsupported_context_graph_query",
                meta={
                    "include": request.include,
                    "fallbacks": list(plan.planner_fallbacks),
                },
            )

        outcomes: list[LegOutcome] = []
        for leg in plan.legs:
            executor = self._executors.get(leg.family)
            if executor is None:
                outcomes.append(
                    LegOutcome(
                        name=leg.name,
                        family=leg.family,
                        strategy=leg.strategy.value,
                        error="no_executor",
                        fallback_reason="unsupported_family",
                        compat=leg.compat,
                    )
                )
                continue
            try:
                outcome = executor(leg, request)
            except Exception as exc:  # noqa: BLE001
                outcome = LegOutcome(
                    name=leg.name,
                    family=leg.family,
                    strategy=leg.strategy.value,
                    error=str(exc) or exc.__class__.__name__,
                    fallback_reason="executor_error",
                    compat=leg.compat,
                )
            outcomes.append(outcome)

        return self._merge(plan, request, outcomes)

    def _merge(
        self,
        plan: ExecutionPlan,
        request: ContextGraphQuery,
        outcomes: list[LegOutcome],
    ) -> ContextGraphResult:
        fallbacks: list[dict[str, Any]] = list(plan.planner_fallbacks)
        for outcome in outcomes:
            if outcome.error or outcome.fallback_reason:
                fallbacks.append(
                    {
                        "family": outcome.family,
                        "reason": outcome.fallback_reason or "executor_error",
                        "detail": outcome.error,
                    }
                )

        legs_meta = [
            {
                "name": o.name,
                "family": o.family,
                "strategy": o.strategy,
                "count": o.count,
                "compat": o.compat,
                "error": o.error,
            }
            for o in outcomes
        ]

        if plan.merge_policy == MergePolicy.SINGLE:
            outcome = outcomes[0]
            meta: dict[str, Any] = {"legs": legs_meta}
            if fallbacks:
                meta["fallbacks"] = fallbacks
            if outcome.compat:
                meta["compat"] = True
            return ContextGraphResult(
                kind=outcome.family,
                goal=request.goal.value,
                strategy=request.strategy.value,
                result=outcome.result,
                error=outcome.error,
                meta=meta,
            )

        results_by_family: dict[str, Any] = {}
        for outcome in outcomes:
            if outcome.error is not None:
                continue
            results_by_family[outcome.family] = outcome.result

        meta = {"legs": legs_meta, "merge": "multi"}
        if fallbacks:
            meta["fallbacks"] = fallbacks
        return ContextGraphResult(
            kind="multi",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=results_by_family,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Per-family executors
    # ------------------------------------------------------------------
    def _exec_semantic_search(
        self, leg: QueryLeg, request: ContextGraphQuery
    ) -> LegOutcome:
        rows = search_pot_context(
            self._episodic,
            request.pot_id,
            request.query or "",
            limit=leg.limit,
            node_labels=request.node_labels or None,
            repo_name=request.scope.repo_name,
            source_description=_first(request.source_descriptions),
            include_invalidated=request.include_invalidated,
            as_of=leg.as_of,
            episode_uuid=_first(request.episode_uuids),
            structural=self._structural,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )

    def _exec_change_history(
        self, leg: QueryLeg, request: ContextGraphQuery
    ) -> LegOutcome:
        rows = get_change_history(
            self._structural,
            request.pot_id,
            function_name=request.scope.function_name,
            file_path=request.scope.file_path,
            limit=leg.limit,
            repo_name=request.scope.repo_name,
            pr_number=request.scope.pr_number,
            as_of=leg.as_of,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )

    def _exec_owners(self, leg: QueryLeg, request: ContextGraphQuery) -> LegOutcome:
        rows = get_file_owners(
            self._structural,
            request.pot_id,
            request.scope.file_path or "",
            limit=leg.limit,
            repo_name=request.scope.repo_name,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )

    def _exec_decisions(self, leg: QueryLeg, request: ContextGraphQuery) -> LegOutcome:
        rows = get_decisions(
            self._structural,
            request.pot_id,
            file_path=request.scope.file_path,
            function_name=request.scope.function_name,
            limit=leg.limit,
            repo_name=request.scope.repo_name,
            pr_number=request.scope.pr_number,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
        )

    def _exec_pr_review_context(
        self, leg: QueryLeg, request: ContextGraphQuery
    ) -> LegOutcome:
        out = get_pr_review_context(
            self._structural,
            request.pot_id,
            request.scope.pr_number or 0,
            repo_name=request.scope.repo_name,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=out,
        )

    def _exec_pr_diff(self, leg: QueryLeg, request: ContextGraphQuery) -> LegOutcome:
        rows = get_pr_diff(
            self._structural,
            request.pot_id,
            request.scope.pr_number or 0,
            file_path=request.scope.file_path,
            limit=leg.limit,
            repo_name=request.scope.repo_name,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
            compat=True,
        )

    def _exec_project_graph(
        self, leg: QueryLeg, request: ContextGraphQuery
    ) -> LegOutcome:
        out = get_project_graph(
            self._structural,
            request.pot_id,
            pr_number=request.scope.pr_number,
            limit=leg.limit,
            scope={
                "repo_name": request.scope.repo_name,
                "services": request.scope.services,
                "features": request.scope.features,
                "environment": request.scope.environment,
                "user": request.scope.user,
            },
            include=request.include,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=out,
        )

    def _exec_graph_overview(
        self, leg: QueryLeg, request: ContextGraphQuery
    ) -> LegOutcome:
        out = get_graph_overview(
            self._structural,
            self._episodic,
            request.pot_id,
            top_entities_limit=leg.limit,
        )
        return LegOutcome(
            name=leg.name,
            family=leg.family,
            strategy=leg.strategy.value,
            result=out,
        )

    async def _answer_async(
        self,
        leg: QueryLeg,
        request: ContextGraphQuery,
        plan: ExecutionPlan,
    ) -> ContextGraphResult:
        if self._resolution_service is None:
            return ContextGraphResult(
                kind="resolve_context",
                goal=request.goal.value,
                strategy=request.strategy.value,
                error="resolution_service_unavailable",
                meta={"legs": [{"name": leg.name, "family": leg.family}]},
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
        meta: dict[str, Any] = {
            "legs": [
                {
                    "name": leg.name,
                    "family": leg.family,
                    "strategy": leg.strategy.value,
                }
            ]
        }
        if plan.planner_fallbacks:
            meta["fallbacks"] = list(plan.planner_fallbacks)
        return ContextGraphResult(
            kind="resolve_context",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=bundle_to_agent_envelope(bundle),
            meta=meta,
        )


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _mode_for_strategy(strategy: ContextGraphStrategy) -> str:
    if strategy == ContextGraphStrategy.HYBRID:
        return "balanced"
    return "fast"
