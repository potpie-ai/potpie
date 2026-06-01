"""``ContextGraphPort`` over one :class:`GraphBackend` — no second graph stack.

One read contract: every read routes through the single :class:`ReadOrchestrator`
(P8/P9) — intent → include families → P9 readers over the canonical claim store →
ranking → one :class:`AgentEnvelope`. There is no server-side answer synthesis and
no agentic read loop: the engine returns ranked evidence and the *agent* reasons
over it (events→answer is the agent's job). ``goal`` survives only as a structural
hint on the request (TIMELINE/RETRIEVE/…); it never selects a different read path.

G1b: this is now a thin DTO-translation facade over a single ``GraphBackend`` — it
no longer owns a private ``GraphWriterPort`` + orchestrator. Reads run the shared
``ReadOrchestrator`` over ``backend.claim_query``; writes go through
``backend.mutation.apply_async``; reset through ``backend.mutation.reset_pot``. The
*same* backend (``Neo4jGraphBackend`` in managed) backs the canonical
``GraphService``, so local and managed share one storage substrate, not two stacks.
"""

from __future__ import annotations

import asyncio
from typing import Any

from application.services.envelope_builder import envelope_to_dict
from application.services.read_orchestrator import ReadOrchestrator
from domain.agent_envelope import AgentEnvelope
from domain.graph_mutations import ProvenanceContext
from domain.graph_query import (
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphScope,
)
from domain.ports.context_graph import ContextGraphPort
from domain.ports.graph.backend import GraphBackend
from domain.reconciliation import ReconciliationPlan, ReconciliationResult


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


class ContextGraphService(ContextGraphPort):
    """``ContextGraphPort`` over one ``GraphBackend`` (reads + writes)."""

    def __init__(self, *, backend: GraphBackend) -> None:
        self._backend = backend
        # The shared read trunk over this backend's canonical claim store — the
        # same construction the local GraphService uses.
        self._orchestrator = ReadOrchestrator(claim_query=backend.claim_query)

    @property
    def enabled(self) -> bool:
        # in_memory/embedded are always available; neo4j reports settings.is_enabled().
        return bool(getattr(self._backend, "enabled", True))

    @property
    def backed_includes(self) -> frozenset[str]:
        """Include families this service's read trunk actually serves; the
        coherence check in the composition root asserts against this."""
        return self._orchestrator.backed_includes

    # ------------------------------------------------------------------
    # Read surface — one trunk, one envelope
    # ------------------------------------------------------------------
    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        return self._orchestrate(request)

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        # Reads are synchronous over the claim store; run off the event loop
        # so async callers (FastAPI handlers, agent tools) don't block.
        return await asyncio.to_thread(self._orchestrate, request)

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
        """Apply a validated reconciliation plan through the backend's mutation port."""
        return await self._backend.mutation.apply_async(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        # The backend mutation owns the sync/async bridge (Neo4j refuses inside a
        # running loop, matching this port's no-async-reset contract).
        inner = self._backend.mutation.reset_pot(pot_id)
        ok = bool(inner.get("ok", True))  # backends with no explicit ok succeed
        out: dict[str, Any] = {"pot_id": pot_id, "ok": ok, "graph_writer": inner}
        if not ok:
            out["error"] = inner.get("error", "graph_reset_failed")
        return out
