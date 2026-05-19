"""Unified context graph adapter backed by Graphiti + canonical Neo4j reads.

Reads route through :class:`ContextReaderRegistry` (Phase 3): the
registry is the only thing that decides which evidence families to
run, runs them, and merges the result envelopes. The adapter no longer
holds per-family executors. Writes remain bounded: ``apply_plan`` for
validated reconciliation plan slices and ``write_raw_episode`` for
direct episode writes.

``goal=ANSWER`` is intentionally not a reader — it composes
``resolve_context`` + the answer synthesizer and stays in this adapter.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from adapters.outbound.graphiti.apply_plan import (
    apply_reconciliation_plan,
    apply_reconciliation_plan_async,
)
from adapters.outbound.graphiti.ingest_episode import ingest_episode as ingest_episode_uc
from adapters.outbound.graphiti.port import EpisodicGraphPort
from adapters.outbound.neo4j.port import StructuralReadPort
from adapters.outbound.reconciliation.context_graph_tools import READ_TOOL_DESCRIPTORS
from application.services.context_reader_registry import ContextReaderRegistry
from application.use_cases.resolve_context import resolve_context
from domain.actor import Actor
from domain.agent_context_port import bundle_to_agent_envelope
from domain.graph_mutations import ProvenanceContext
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphStrategy,
    preset_change_history,
    preset_file_owners,
    preset_graph_overview,
    preset_semantic_search,
)
from domain.intelligence_models import (
    ArtifactRef,
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)
from domain.ports.answer_synthesizer import AnswerSynthesizerPort
from domain.ports.context_graph import ContextGraphPort
from domain.ports.query_agent import QueryAgentPort, QueryAgentResult
from domain.reconciliation import ReconciliationPlan, ReconciliationResult

_AGENTIC_GOALS = (ContextGraphGoal.ANSWER, ContextGraphGoal.INVESTIGATE)


class GraphitiContextGraphAdapter(ContextGraphPort):
    """One graph query surface over the registered ``ContextReader``s."""

    def __init__(
        self,
        *,
        episodic: EpisodicGraphPort,
        structural: StructuralReadPort,
        readers: ContextReaderRegistry,
        resolution_service: Any | None = None,
        answer_synthesizer: AnswerSynthesizerPort | None = None,
        query_agent: QueryAgentPort | None = None,
    ) -> None:
        self._episodic = episodic
        self._structural = structural
        self._readers = readers
        self._resolution_service = resolution_service
        self._answer_synthesizer = answer_synthesizer
        self._query_agent = query_agent

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._episodic, "enabled", False))

    @property
    def readers(self) -> ContextReaderRegistry:
        return self._readers

    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal in _AGENTIC_GOALS:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.query_async(request))
            raise RuntimeError(
                "GraphitiContextGraphAdapter.query() cannot execute answer or "
                "investigate queries while an event loop is already running; "
                "use query_async()."
            )
        return self._readers.execute(request)

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal == ContextGraphGoal.ANSWER:
            return await self._answer_async(request)
        if request.goal == ContextGraphGoal.INVESTIGATE:
            return await self._investigate_async(request)
        return self._readers.execute(request)

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        return apply_reconciliation_plan(
            self._episodic,
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    async def apply_plan_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        """Async-native plan apply — use from agent tools / async handlers."""
        return await apply_reconciliation_plan_async(
            self._episodic,
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        *,
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        return ingest_episode_uc(
            self._episodic,
            pot_id,
            name,
            episode_body,
            source_description,
            reference_time,
            actor=actor,
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"pot_id": pot_id, "ok": False}

        ep = self._episodic.reset_pot(pot_id)
        out["episodic"] = ep
        if not ep.get("ok"):
            out["error"] = ep.get("error", "episodic_reset_failed")
            return out

        out["ok"] = True
        return out

    # ------------------------------------------------------------------
    # Answer path (resolve_context + synthesizer) — not a reader.
    # ------------------------------------------------------------------
    async def _answer_async(
        self, request: ContextGraphQuery
    ) -> ContextGraphResult:
        if self._resolution_service is None:
            return ContextGraphResult(
                kind="resolve_context",
                goal=request.goal.value,
                strategy=request.strategy.value,
                error="resolution_service_unavailable",
                meta={"path": "answer"},
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
        synthesized: str | None = None
        synthesis_usage: dict[str, Any] | None = None
        if self._answer_synthesizer is not None:
            synthesized = await self._answer_synthesizer.synthesize(bundle)
            synthesis_usage = getattr(self._answer_synthesizer, "last_usage", None)
        meta: dict[str, Any] = {
            "path": "answer",
            "answer_summary_source": "synthesized" if synthesized else "counts",
        }
        return ContextGraphResult(
            kind="resolve_context",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=bundle_to_agent_envelope(
                bundle,
                answer_summary=synthesized,
                synthesis_usage=synthesis_usage,
            ),
            meta=meta,
        )


    # ------------------------------------------------------------------
    # Investigate path (agentic read loop) — not a reader.
    # Reuses the same 4 read tools as the reconciliation agent; falls back
    # to the deterministic answer path when the agent is unavailable.
    # ------------------------------------------------------------------
    async def _investigate_async(
        self, request: ContextGraphQuery
    ) -> ContextGraphResult:
        if self._query_agent is None:
            return await self._answer_fallback(request, "query_agent_not_configured")

        repo_name = request.scope.repo_name

        def _build_preset(name: str, args: dict[str, Any]) -> ContextGraphQuery | dict:
            if name == "context_search":
                q = str(args.get("query") or "").strip()
                if not q:
                    return {"error": "query_required", "kind": "error"}
                labels = args.get("node_labels") or []
                return preset_semantic_search(
                    pot_id=request.pot_id,
                    query=q,
                    limit=max(1, min(int(args.get("limit") or 8), 25)),
                    repo_name=repo_name,
                    node_labels=[str(x) for x in labels] if isinstance(labels, list) else None,
                )
            if name == "context_recent_changes":
                file_path = args.get("file_path")
                function_name = args.get("function_name")
                pr_number = args.get("pr_number")
                if not any([file_path, function_name, pr_number]):
                    return {
                        "error": "one_of_file_path_function_name_pr_number_required",
                        "kind": "error",
                    }
                return preset_change_history(
                    pot_id=request.pot_id,
                    file_path=file_path,
                    function_name=function_name,
                    pr_number=int(pr_number) if pr_number else None,
                    repo_name=repo_name,
                    limit=max(1, min(int(args.get("limit") or 10), 25)),
                )
            if name == "context_file_owners":
                file_path = args.get("file_path")
                if not file_path:
                    return {"error": "file_path_required", "kind": "error"}
                return preset_file_owners(
                    pot_id=request.pot_id,
                    file_path=str(file_path),
                    repo_name=repo_name,
                    limit=max(1, min(int(args.get("limit") or 5), 10)),
                )
            if name == "context_graph_overview":
                return preset_graph_overview(
                    pot_id=request.pot_id,
                    limit=max(1, min(int(args.get("limit") or 20), 50)),
                )
            return {"error": f"unknown_tool:{name}", "kind": "error"}

        async def _run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            preset = _build_preset(name, args)
            if isinstance(preset, dict):
                return preset
            result = await asyncio.to_thread(self._readers.execute, preset)
            return result.model_dump()

        try:
            res = await self._query_agent.investigate(
                request,
                tools=list(READ_TOOL_DESCRIPTORS),
                run_tool=_run_tool,
            )
        except Exception:  # noqa: BLE001 - any agent failure degrades gracefully
            return await self._answer_fallback(request, "query_agent_error")

        if res is None:
            return await self._answer_fallback(request, "query_agent_unavailable")

        return ContextGraphResult(
            kind="query_agent",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=_investigate_envelope(res),
            meta={
                "path": "investigate",
                "iterations": res.iterations,
                "cost": {"query_agent": res.usage} if res.usage else {},
            },
        )

    async def _answer_fallback(
        self, request: ContextGraphQuery, reason: str
    ) -> ContextGraphResult:
        """Degrade an ``investigate`` request onto the deterministic answer path."""
        result = await self._answer_async(request)
        meta = dict(result.meta or {})
        meta["path"] = "investigate_fallback"
        meta["fallback"] = reason
        return result.model_copy(update={"meta": meta})


def _investigate_envelope(res: QueryAgentResult) -> dict[str, Any]:
    """Answer-envelope-shaped superset so existing clients render it, plus
    an ``agent`` block carrying the tool trace.
    """
    has_evidence = bool(res.evidence)
    confidence = (
        res.confidence
        if res.confidence is not None
        else (0.7 if has_evidence else 0.2)
    )
    return {
        "ok": True,
        "answer": {"summary": res.answer},
        "evidence": res.evidence,
        "source_refs": res.source_refs,
        "confidence": confidence,
        "coverage": {
            "status": "complete" if has_evidence else "empty",
            "available": sorted({s.tool for s in res.steps}),
            "missing": [],
        },
        "quality": {"status": "ok" if has_evidence else "watch"},
        "fallbacks": [],
        "agent": {
            "iterations": res.iterations,
            "steps": [
                {
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "result_kind": s.result_kind,
                    "result_count": s.result_count,
                }
                for s in res.steps
            ],
            "usage": res.usage,
        },
        "meta": {"cost": {"query_agent": res.usage} if res.usage else {}},
    }


def _mode_for_strategy(strategy: ContextGraphStrategy) -> str:
    if strategy == ContextGraphStrategy.HYBRID:
        return "balanced"
    return "fast"
