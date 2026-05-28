"""Unified context graph adapter over the single graph writer + reader trunk.

Reads route through the single :class:`ReadOrchestrator` (P8/P9): intent →
include families → P9 readers over the canonical claim store → ranking →
one :class:`AgentEnvelope`. Writes go straight through
:class:`GraphWriterPort` via ``apply_reconciliation_plan``.

``goal=ANSWER`` composes the resolve envelope + the answer synthesizer.
``goal=INVESTIGATE`` runs the agentic read loop (each tool resolves through the
same orchestrator) and degrades to the answer path when no agent is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from adapters.outbound.graph.apply_plan import apply_reconciliation_plan
from adapters.outbound.graph.neo4j_writer import GraphWriterPort
from adapters.outbound.reconciliation.context_graph_tools import (
    READ_TOOL_DESCRIPTORS,
    READ_TOOL_INCLUDE,
)
from application.services.envelope_builder import envelope_to_dict
from application.services.read_orchestrator import ReadOrchestrator
from domain.agent_envelope import AgentEnvelope
from domain.graph_mutations import ProvenanceContext
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphScope,
)
from domain.ports.answer_synthesizer import AnswerSynthesizerPort
from domain.ports.context_graph import ContextGraphPort
from domain.ports.query_agent import QueryAgentPort, QueryAgentResult
from domain.reconciliation import ReconciliationPlan, ReconciliationResult

_AGENTIC_GOALS = (ContextGraphGoal.ANSWER, ContextGraphGoal.INVESTIGATE)


def _scope_to_dict(scope: ContextGraphScope) -> dict[str, Any]:
    """Flatten the agent scope into the keys the P9 readers understand."""
    return {
        "repo": scope.repo_name,
        "repo_name": scope.repo_name,
        "branch": scope.branch,
        "file_path": scope.file_path,
        "function_name": scope.function_name,
        "symbol": scope.symbol,
        "pr_number": scope.pr_number,
        "service": scope.services[0] if scope.services else None,
        "services": list(scope.services),
        "features": list(scope.features),
        "environment": scope.environment,
        "ticket_ids": list(scope.ticket_ids),
        "user": scope.user,
    }


def _fallback_summary(env: AgentEnvelope) -> str:
    """Deterministic answer summary when no LLM synthesizer is available."""
    if not env.items:
        return "No project context found for this query."
    by_family: dict[str, int] = {}
    for item in env.items:
        by_family[item.include] = by_family.get(item.include, 0) + 1
    parts = [f"{n} {fam}" for fam, n in sorted(by_family.items())]
    return "Resolved " + ", ".join(parts) + " for this request."


class ContextGraphService(ContextGraphPort):
    """One graph query surface over the single read orchestrator + writer."""

    def __init__(
        self,
        *,
        graph_writer: GraphWriterPort,
        orchestrator: ReadOrchestrator,
        answer_synthesizer: AnswerSynthesizerPort | None = None,
        query_agent: QueryAgentPort | None = None,
    ) -> None:
        self._graph_writer = graph_writer
        self._orchestrator = orchestrator
        self._answer_synthesizer = answer_synthesizer
        self._query_agent = query_agent

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._graph_writer, "enabled", False))

    # ------------------------------------------------------------------
    # Read surface — one trunk
    # ------------------------------------------------------------------
    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal in _AGENTIC_GOALS:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.query_async(request))
            raise RuntimeError(
                "ContextGraphService.query() cannot execute answer or "
                "investigate queries while an event loop is already running; "
                "use query_async()."
            )
        return self._orchestrate(request)

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        if request.goal == ContextGraphGoal.ANSWER:
            return await self._answer_async(request)
        if request.goal == ContextGraphGoal.INVESTIGATE:
            return await self._investigate_async(request)
        return self._orchestrate(request)

    def _resolve_envelope(self, request: ContextGraphQuery) -> AgentEnvelope:
        return self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent=request.intent,
            query=request.query,
            scope=_scope_to_dict(request.scope),
            include=list(request.include),
            exclude=list(request.exclude),
            as_of=request.as_of,
            since=request.since,
            until=request.until,
            max_items=request.budget.max_items,
            include_invalidated=request.include_invalidated,
            metadata={"query": request.query or ""},
        )

    def _orchestrate(self, request: ContextGraphQuery) -> ContextGraphResult:
        env = self._resolve_envelope(request)
        return ContextGraphResult(
            kind="resolve",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=envelope_to_dict(env),
            meta={"path": "resolve"},
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.apply_plan_async(
                    plan,
                    expected_pot_id=expected_pot_id,
                    provenance_context=provenance_context,
                )
            )
        raise RuntimeError(
            "ContextGraphService.apply_plan() cannot run inside an "
            "event loop; use apply_plan_async()."
        )

    async def apply_plan_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        """Apply a validated reconciliation plan through the single graph writer."""
        return await apply_reconciliation_plan(
            self._graph_writer,
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            inner = asyncio.run(self._graph_writer.reset_pot(pot_id))
        else:
            raise RuntimeError(
                "ContextGraphService.reset_pot() cannot run inside "
                "an event loop; await graph_writer.reset_pot(...) instead."
            )
        out: dict[str, Any] = {
            "pot_id": pot_id,
            "ok": bool(inner.get("ok")),
            "graph_writer": inner,
        }
        if not inner.get("ok"):
            out["error"] = inner.get("error", "graph_writer_reset_failed")
        return out

    # ------------------------------------------------------------------
    # Answer path (resolve + synthesizer)
    # ------------------------------------------------------------------
    async def _answer_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        env = await asyncio.to_thread(self._resolve_envelope, request)
        synthesized: str | None = None
        synthesis_usage: dict[str, Any] | None = None
        if self._answer_synthesizer is not None:
            synthesized = await self._answer_synthesizer.synthesize(env)
            synthesis_usage = getattr(self._answer_synthesizer, "last_usage", None)
        result = envelope_to_dict(env)
        result["answer"] = {"summary": synthesized or _fallback_summary(env)}
        meta: dict[str, Any] = {
            "path": "answer",
            "answer_summary_source": "synthesized" if synthesized else "fallback",
        }
        if synthesis_usage is not None:
            meta["cost"] = {"synthesis": synthesis_usage}
        return ContextGraphResult(
            kind="resolve_context",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=result,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Investigate path (agentic read loop) — every read tool resolves
    # through the same orchestrator; degrades to the answer path.
    # ------------------------------------------------------------------
    async def _investigate_async(
        self, request: ContextGraphQuery
    ) -> ContextGraphResult:
        if self._query_agent is None:
            return await self._answer_fallback(request, "query_agent_not_configured")

        async def _run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            q = str(args.get("query") or request.query or "").strip()
            update: dict[str, Any] = {
                "query": q or request.query,
                "goal": ContextGraphGoal.RETRIEVE,
            }
            # Honor the tool the agent called: a targeted tool pins its reader
            # include; the generic ``context_search`` (include None) keeps the
            # request's intent-driven routing.
            include = READ_TOOL_INCLUDE.get(name)
            if include is not None:
                update["include"] = [include]
            # Carry scoped args (file / function / PR) so e.g. context_timeline
            # narrows correctly.
            scope_updates: dict[str, Any] = {}
            for key in ("file_path", "function_name"):
                if args.get(key):
                    scope_updates[key] = str(args[key])
            if args.get("pr_number"):
                try:
                    scope_updates["pr_number"] = int(args["pr_number"])
                except (TypeError, ValueError):
                    pass
            if scope_updates:
                update["scope"] = request.scope.model_copy(update=scope_updates)
            sub = request.model_copy(update=update)
            return await asyncio.to_thread(lambda: self._orchestrate(sub).result)

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
