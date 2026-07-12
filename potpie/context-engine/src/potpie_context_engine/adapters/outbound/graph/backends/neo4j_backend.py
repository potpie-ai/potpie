"""Neo4j ``GraphBackend`` — the shape-first production target.

This is the canonical-store profile for the migration. The backend assembles
it from capability adapters and delegates the two *source-of-truth* ports to the
existing, battle-tested Neo4j code; semantic search is backed by Neo4j's native
relationship vector index, while inspection/snapshot remain fail-closed stubs
until they are built out.

    claim_query  -> Neo4jClaimQueryStore           (existing, real)
    mutation     -> existing apply path             # TODO(stage-N)
    analytics    -> ClaimQueryAnalytics             (computed from claim_query, real)
    semantic     -> ClaimQuerySemanticSearch        (native vector via claim_query)
    inspection   -> CapabilityNotImplemented        # TODO(stage-N): cypher traversal
    snapshot     -> CapabilityNotImplemented        # TODO(stage-N): portable export/import

Neo4j imports are lazy so the skeleton (and the in_memory profile) load without
the ``graph`` extra installed; a missing driver surfaces only when this profile
is actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Coroutine, Mapping, Sequence, TypeVar

from potpie_context_engine.adapters.outbound.graph.backends._unimplemented import (
    UnimplementedInspection,
    UnimplementedSnapshot,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.cypher import _coerce_props_for_neo4j
from potpie_context_engine.adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_REPAIR_LIMIT,
    ENTITY_SUMMARY_SCAN_CYPHER,
    ENTITY_SUMMARY_UPDATE_CYPHER,
    repaired_entity_properties,
)
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_engine.domain.graph_mutations import ProvenanceContext
from potpie_context_engine.domain.lifecycle import SetupPlan, StepResult
from potpie_context_engine.domain.ports.claim_query import ClaimQueryPort
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.graph.analytics import GraphAnalyticsPort
from potpie_context_engine.domain.ports.graph.backend import BackendCapabilities
from potpie_context_engine.domain.ports.graph.inspection import GraphInspectionPort
from potpie_context_engine.domain.ports.graph.mutation import (
    BackendReadiness,
    GraphMutationPort,
)
from potpie_context_engine.domain.ports.graph.semantic import SemanticSearchPort
from potpie_context_engine.domain.ports.graph.snapshot import GraphSnapshotPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort
from potpie_context_engine.domain.reconciliation import MutationBatch, MutationResult

_PROFILE = "neo4j"
_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
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
    """``GraphMutationPort`` over ``Neo4jGraphWriter`` + ``apply_mutation_batch``.

    ``apply_async`` is the native door (it ``await``s the async writer directly);
    ``apply`` is a loop-aware sync bridge for CLI/tests. The writer is created
    once and reused — its async driver binds to the loop that first ``await``s
    it, which in managed is uvicorn's single request loop (the same pattern the
    production ``ContextGraphService`` uses with one long-lived writer).
    """

    settings: ContextEngineSettingsPort
    writer: GraphWriterPort | None = None  # injected or lazily created on first use
    embedder: EmbedderPort | None = None

    def _get_writer(self) -> GraphWriterPort:
        if self.writer is None:
            from potpie_context_engine.adapters.outbound.graph import Neo4jGraphWriter

            self.writer = Neo4jGraphWriter(self.settings, embedder=self.embedder)
        return self.writer

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.apply_plan import (
            apply_mutation_batch,
        )

        return await apply_mutation_batch(
            self._get_writer(),
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        return _run_sync(
            self.apply_async(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
            )
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Sequence[str], reason: str | None = None
    ) -> int:
        # TODO(stage-N): cypher invalidation by claim key.
        from potpie_context_engine.domain.errors import CapabilityNotImplemented

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
            detail="neo4j claim_query + mutation + semantic + analytics wired; inspection/snapshot pending",
            capability_ready={
                "mutation": True,
                "claim_query": True,
                "analytics": True,
                "semantic": True,
                "inspection": False,
                "snapshot": False,
            },
        )


@dataclass(slots=True)
class Neo4jGraphBackend:
    """Neo4j-backed ``GraphBackend`` (shape-first; projections are TODO)."""

    settings: ContextEngineSettingsPort
    writer: GraphWriterPort | None = None  # optional shared writer; reused by mutation
    embedder: EmbedderPort | None = None
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _Neo4jMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        # Lazy: only touch neo4j when this profile is selected.
        from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
            Neo4jClaimQueryStore,
        )

        self._claim_query = Neo4jClaimQueryStore(self.settings, embedder=self.embedder)
        self._mutation = _Neo4jMutation(
            self.settings, writer=self.writer, embedder=self.embedder
        )
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)

    @property
    def enabled(self) -> bool:
        # Cheap config probe (no driver build): graph availability for policy and
        # ContextGraphService.enabled. Mirrors Neo4jGraphWriter.enabled.
        return self.settings.is_enabled()

    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def graph_writer(self) -> GraphWriterPort:
        """Compatibility alias for old ingestion paths that seed via writer."""
        return self._mutation._get_writer()

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> GraphMutationPort:
        return self._mutation

    @property
    def semantic(self) -> SemanticSearchPort:
        return self._semantic

    @property
    def inspection(self) -> GraphInspectionPort:
        return UnimplementedInspection(_PROFILE)

    @property
    def analytics(self) -> GraphAnalyticsPort:
        # Real: counts/freshness/quality are computed from the canonical
        # claim store, which this profile already serves.
        return ClaimQueryAnalytics(
            self._claim_query,
            entity_summary_repair=self._repair_entity_summaries,
        )

    @property
    def snapshot(self) -> GraphSnapshotPort:
        return UnimplementedSnapshot(_PROFILE)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=True,
            inspection=False,
            snapshot=False,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        from potpie_context_engine.domain.lifecycle import DONE, FAILED

        if not self.enabled:
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail="neo4j backend is not configured or context graph is disabled",
                metadata={"profile": _PROFILE},
            )
        try:
            ok = bool(_run_sync(self.graph_writer.ensure_indexes()))
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=str(exc),
                metadata={"profile": _PROFILE},
            )
        return StepResult(
            step="backend.provision",
            state=DONE if ok else FAILED,
            detail="neo4j backend ready" if ok else "neo4j index setup failed",
            metadata={"profile": _PROFILE},
        )

    def _repair_entity_summaries(self, pot_id: str) -> int:
        from neo4j import GraphDatabase

        uri = self.settings.neo4j_uri()
        user = self.settings.neo4j_user()
        password = self.settings.neo4j_password()
        if not uri or user is None or password is None:
            raise RuntimeError("neo4j_unavailable")

        repaired = 0
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                rows = list(
                    session.run(
                        ENTITY_SUMMARY_SCAN_CYPHER,
                        gid=pot_id,
                        limit=ENTITY_SUMMARY_REPAIR_LIMIT,
                    )
                )
                for row in rows:
                    key = str(row.get("key") or "").strip()
                    if not key:
                        continue
                    raw_props = row.get("props")
                    fixed = repaired_entity_properties(
                        key, raw_props if isinstance(raw_props, Mapping) else {}
                    )
                    if fixed is None:
                        continue
                    result = session.run(
                        ENTITY_SUMMARY_UPDATE_CYPHER,
                        gid=pot_id,
                        key=key,
                        props=_coerce_props_for_neo4j(fixed),
                    )
                    rec = result.single()
                    result.consume()
                    repaired += int(rec["cnt"]) if rec is not None else 0
        finally:
            driver.close()
        return repaired


__all__ = ["Neo4jGraphBackend"]
