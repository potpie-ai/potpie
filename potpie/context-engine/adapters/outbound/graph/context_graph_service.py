"""Legacy ``ContextGraphPort`` compatibility over canonical ``GraphService``.

The canonical graph service owns the read trunk and semantic write path. This
module exists only to keep older managed callers compiling while they migrate to
``GraphService`` / ``AgentContextPort`` DTOs. It must not construct readers or
apply old reconciliation plans directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from application.services.envelope_builder import envelope_to_dict
from application.services.graph_service import DefaultGraphService
from domain.agent_envelope import AgentEnvelope
from domain.errors import CapabilityNotImplemented
from domain.graph_mutations import ProvenanceContext
from domain.graph_query import (
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphScope,
)
from domain.ports.agent_context import ResolveRequest
from domain.ports.context_graph import ContextGraphPort
from domain.ports.graph.backend import GraphBackend
from domain.ports.services.graph_service import GraphService
from domain.reconciliation import MutationBatch, MutationResult


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
        "source_refs": list(scope.source_refs),
    }


class ContextGraphService(ContextGraphPort):
    """Legacy DTO translator over canonical ``GraphService``."""

    def __init__(
        self,
        *,
        graph: GraphService | None = None,
        backend: GraphBackend | None = None,
    ) -> None:
        if graph is None:
            if backend is None:
                raise TypeError("ContextGraphService requires graph= or backend=")
            # Back-compat for older tests/callers. The resulting service is still
            # canonical: ``DefaultGraphService`` owns readers and write lowering.
            graph = DefaultGraphService(backend=backend)
        self._graph = graph

    @property
    def enabled(self) -> bool:
        backend = getattr(self._graph, "backend", None)
        return bool(getattr(backend, "enabled", True))

    @property
    def backed_includes(self) -> frozenset[str]:
        """Include families this service's read trunk actually serves; the
        coherence check in the composition root asserts against this."""
        return frozenset(getattr(self._graph, "backed_includes", frozenset()))

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
        return self._graph.resolve(
            ResolveRequest(
                pot_id=request.pot_id,
                task=request.query,
                intent=request.intent,
                scope=_scope_to_dict(request.scope),
                include=tuple(request.include),
                exclude=tuple(request.exclude),
                as_of=request.as_of,
                since=request.since,
                until=request.until,
                max_items=request.budget.max_items,
                include_invalidated=request.include_invalidated,
                freshness_preference=request.budget.freshness,
                metadata={
                    "legacy_context_graph_query": True,
                    "query": request.query or "",
                    "goal": request.goal.value,
                    "strategy": request.strategy.value,
                    "consumer_hint": request.consumer_hint,
                },
            )
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
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        del plan, expected_pot_id, provenance_context
        raise CapabilityNotImplemented(
            "context_graph.apply_plan",
            detail=(
                "legacy MutationBatch apply is disabled; use GraphService.mutate "
                "or context_record so writes go through semantic validation."
            ),
            recommended_next_action="Route writes through GraphService.mutate.",
        )

    async def apply_plan_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        del plan, expected_pot_id, provenance_context
        raise CapabilityNotImplemented(
            "context_graph.apply_plan_async",
            detail=(
                "legacy MutationBatch apply is disabled; use GraphService.mutate "
                "or context_record so writes go through semantic validation."
            ),
            recommended_next_action="Route writes through GraphService.mutate.",
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        # The backend mutation owns the sync/async bridge (Neo4j refuses inside a
        # running loop, matching this port's no-async-reset contract).
        backend = getattr(self._graph, "backend", None)
        if backend is None:
            raise CapabilityNotImplemented(
                "context_graph.reset_pot",
                detail="the wrapped graph service does not expose a backend",
            )
        inner = backend.mutation.reset_pot(pot_id)
        ok = bool(inner.get("ok", True))  # backends with no explicit ok succeed
        out: dict[str, Any] = {"pot_id": pot_id, "ok": ok, "graph_writer": inner}
        if not ok:
            out["error"] = inner.get("error", "graph_reset_failed")
        return out
