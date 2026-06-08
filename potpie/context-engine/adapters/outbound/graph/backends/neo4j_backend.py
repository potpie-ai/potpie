"""Neo4j ``GraphBackend`` — the shape-first production target.

This is the canonical-store profile for the migration. The skeleton assembles
it from capability adapters and delegates the two *source-of-truth* ports to the
existing, battle-tested Neo4j code; the four rebuildable-projection ports are
wired to fail-closed stubs until they are built out.

    claim_query  -> Neo4jClaimQueryStore           (existing, real)
    mutation     -> existing apply path             # TODO(stage-N)
    analytics    -> ClaimQueryAnalytics             (computed from claim_query, real)
    semantic     -> CapabilityNotImplemented        # TODO(stage-N): fold neo4j vector
    inspection   -> CapabilityNotImplemented        # TODO(stage-N): cypher traversal
    snapshot     -> CapabilityNotImplemented        # TODO(stage-N): portable export/import

Neo4j imports are lazy so the skeleton (and the in_memory profile) load without
the ``graph`` extra installed; a missing driver surfaces only when this profile
is actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adapters.outbound.graph.backends._unimplemented import (
    UnimplementedInspection,
    UnimplementedSemantic,
    UnimplementedSnapshot,
)
from adapters.outbound.graph.backends.claim_query_analytics import ClaimQueryAnalytics
from domain.graph_mutations import ProvenanceContext
from domain.lifecycle import SetupPlan, StepResult
from domain.ports.claim_query import ClaimQueryPort
from domain.ports.graph.backend import BackendCapabilities
from domain.ports.graph.mutation import BackendReadiness
from domain.reconciliation import ReconciliationPlan, ReconciliationResult

_PROFILE = "neo4j"


def _run_sync(coro: Any) -> Any:
    """Drive a coroutine from a *sync* port entry (CLI/tests).

    Loop-aware: outside a running loop we run it with ``asyncio.run``; inside one
    we refuse rather than bind the writer's async connection pool to a throwaway
    loop and corrupt it — async callers must use the ``*_async`` door. Mirrors
    ``ContextGraphService.apply_plan``.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "Neo4jGraphBackend sync mutation cannot run inside an event loop; "
        "use the async door (mutation.apply_async)."
    )


@dataclass(slots=True)
class _Neo4jMutation:
    """``GraphMutationPort`` over ``Neo4jGraphWriter`` + ``apply_reconciliation_plan``.

    ``apply_async`` is the native door (it ``await``s the async writer directly);
    ``apply`` is a loop-aware sync bridge for CLI/tests. The writer is created
    once and reused — its async driver binds to the loop that first ``await``s
    it, which in managed is uvicorn's single request loop (the same pattern the
    production ``ContextGraphService`` uses with one long-lived writer).
    """

    settings: Any
    writer: Any = None  # injected (shared) or lazily created on first use

    def _get_writer(self) -> Any:
        if self.writer is None:
            from adapters.outbound.graph import Neo4jGraphWriter

            self.writer = Neo4jGraphWriter(self.settings)
        return self.writer

    async def apply_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        from adapters.outbound.graph.apply_plan import apply_reconciliation_plan

        return await apply_reconciliation_plan(
            self._get_writer(),
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def apply(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        return _run_sync(
            self.apply_async(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
            )
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Any, reason: str | None = None
    ) -> int:
        # TODO(stage-N): cypher invalidation by claim key.
        from domain.errors import CapabilityNotImplemented

        raise CapabilityNotImplemented(
            "graph.neo4j.mutation.invalidate",
            recommended_next_action="implement cypher invalidation",
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        # TODO(stage-N): route through the existing hard_reset_pot use case.
        return _run_sync(self._get_writer().reset_pot(pot_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        return BackendReadiness(
            profile=_PROFILE,
            ready=True,
            detail="neo4j claim_query + mutation + analytics wired; semantic/inspection/snapshot pending",
            capability_ready={
                "mutation": True,
                "claim_query": True,
                "analytics": True,
                "semantic": False,
                "inspection": False,
                "snapshot": False,
            },
        )


@dataclass(slots=True)
class Neo4jGraphBackend:
    """Neo4j-backed ``GraphBackend`` (shape-first; projections are TODO)."""

    settings: Any
    writer: Any = None  # optional shared Neo4jGraphWriter; reused by the mutation
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _Neo4jMutation = field(init=False)

    def __post_init__(self) -> None:
        # Lazy: only touch neo4j when this profile is selected.
        from adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore

        self._claim_query = Neo4jClaimQueryStore(self.settings)
        self._mutation = _Neo4jMutation(self.settings, writer=self.writer)

    @property
    def enabled(self) -> bool:
        # Cheap config probe (no driver build): graph availability for policy and
        # ContextGraphService.enabled. Mirrors Neo4jGraphWriter.enabled.
        is_enabled = getattr(self.settings, "is_enabled", None)
        return bool(is_enabled()) if callable(is_enabled) else True

    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _Neo4jMutation:
        return self._mutation

    @property
    def semantic(self) -> UnimplementedSemantic:
        return UnimplementedSemantic(_PROFILE)

    @property
    def inspection(self) -> UnimplementedInspection:
        return UnimplementedInspection(_PROFILE)

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        # Real: counts/freshness/quality are computed from the canonical
        # claim store, which this profile already serves.
        return ClaimQueryAnalytics(self._claim_query)

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(_PROFILE)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=False,
            inspection=False,
            snapshot=False,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        from domain.errors import CapabilityNotImplemented

        raise CapabilityNotImplemented(
            "graph.neo4j.provision",
            detail="neo4j store provisioning (database create, indexes, native vector index) not implemented",
            recommended_next_action="provision neo4j out-of-band, or run 'potpie setup --backend embedded'",
        )


__all__ = ["Neo4jGraphBackend"]
