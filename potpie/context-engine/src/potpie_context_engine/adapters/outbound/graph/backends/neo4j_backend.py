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

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping
import uuid

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionRegistry,
)
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
from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_REPAIR_LIMIT,
    ENTITY_SUMMARY_SCAN_CYPHER,
    ENTITY_SUMMARY_UPDATE_CYPHER,
    repaired_entity_properties,
)
from potpie_context_engine.adapters.outbound.graph.entity_label_repair import (
    ENTITY_LABEL_REPAIR_LIMIT,
    ENTITY_LABEL_SCAN_CYPHER,
    canonical_label_changes,
    repaired_entity_labels,
)
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.ports.claim_query import ClaimQueryPort
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.mutation import BackendReadiness
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

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
    """``GraphMutationPort`` over ``Neo4jGraphWriter`` + ``apply_mutation_batch``.

    ``apply_async`` is the native door (it ``await``s the async writer directly);
    ``apply`` is a loop-aware sync bridge for CLI/tests. The writer is created
    once and reused — its async driver binds to the loop that first ``await``s
    it, which in managed is uvicorn's single request loop (the same pattern the
    production ``ContextGraphService`` uses with one long-lived writer).
    """

    settings: Any
    writer: Any = None  # injected (shared) or lazily created on first use
    embedder: Any = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry
    )

    def _get_writer(self) -> Any:
        if self.writer is None:
            from potpie_context_engine.adapters.outbound.graph import Neo4jGraphWriter

            self.writer = Neo4jGraphWriter(
                self.settings,
                embedder=self.embedder,
                definition=self.definition,
            )
        return self.writer

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.apply_plan import (
            apply_mutation_batch,
        )

        mutation_id = (
            provenance_context.mutation_id
            if provenance_context is not None and provenance_context.mutation_id
            else uuid.uuid4().hex
        )
        context = replace(
            provenance_context or ProvenanceContext(),
            mutation_id=mutation_id,
        )
        return await self.execution_registry.execute_async(
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
            operation=lambda: apply_mutation_batch(
                self._get_writer(),
                deepcopy(plan),
                expected_pot_id=expected_pot_id,
                provenance_context=context,
                definition=self.definition,
                reconciliation_config=reconciliation_config,
            ),
        )

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        lookup = self.execution_registry.lookup(
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
        )
        if lookup.state != MutationExecutionState.absent.value:
            return lookup
        return MutationExecutionLookup(
            state=MutationExecutionState.unsupported.value,
            mutation_id=mutation_id,
            batch_fingerprint=lookup.batch_fingerprint,
            detail="Neo4j mutation receipts are not durable across processes",
        )

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        return _run_sync(
            self.apply_async(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
                reconciliation_config=reconciliation_config,
            )
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Any, reason: str | None = None
    ) -> int:
        # TODO(stage-N): cypher invalidation by claim key.
        from potpie_context_engine.core.errors import CapabilityNotImplemented

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

    settings: Any
    writer: Any = None  # optional shared Neo4jGraphWriter; reused by the mutation
    embedder: Any = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry,
        repr=False,
    )
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _Neo4jMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        # Lazy: only touch neo4j when this profile is selected.
        from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
            Neo4jClaimQueryStore,
        )

        self._claim_query = Neo4jClaimQueryStore(self.settings, embedder=self.embedder)
        writer = self.writer
        bind_writer = getattr(writer, "bind_definition", None)
        if callable(bind_writer):
            writer = bind_writer(self.definition)
        self._mutation = _Neo4jMutation(
            self.settings,
            writer=writer,
            embedder=self.embedder,
            definition=self.definition,
            execution_registry=self.execution_registry,
        )
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)

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
    def graph_writer(self) -> Any:
        """Compatibility alias for old ingestion paths that seed via writer."""
        return self._mutation._get_writer()

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _Neo4jMutation:
        return self._mutation

    @property
    def semantic(self) -> ClaimQuerySemanticSearch:
        return self._semantic

    @property
    def inspection(self) -> UnimplementedInspection:
        return UnimplementedInspection(_PROFILE)

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        # Real: counts/freshness/quality are computed from the canonical
        # claim store, which this profile already serves.
        return ClaimQueryAnalytics(
            self._claim_query,
            entity_summary_repair=self._repair_entity_summaries,
            entity_label_repair=self._repair_entity_labels,
        )

    @property
    def snapshot(self) -> UnimplementedSnapshot:
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

    def bind_definition(self, definition: GraphDefinition) -> Neo4jGraphBackend:
        return replace(self, definition=definition)

    def provision(self) -> BackendProvisionResult:
        if not self.enabled:
            return BackendProvisionResult(
                ok=False,
                detail="neo4j backend is not configured or context graph is disabled",
                metadata={"profile": _PROFILE},
            )
        try:
            ok = bool(_run_sync(self.graph_writer.ensure_indexes()))
        except Exception as exc:  # noqa: BLE001
            return BackendProvisionResult(
                ok=False,
                detail=str(exc),
                metadata={"profile": _PROFILE},
            )
        return BackendProvisionResult(
            ok=ok,
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

    def _repair_entity_labels(self, pot_id: str) -> int:
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
                after = ""
                while True:
                    rows = list(
                        session.run(
                            ENTITY_LABEL_SCAN_CYPHER,
                            gid=pot_id,
                            after=after,
                            limit=ENTITY_LABEL_REPAIR_LIMIT,
                        )
                    )
                    if not rows:
                        break
                    for row in rows:
                        key = str(row.get("key") or "").strip()
                        labels = tuple(row.get("labels") or ())
                        fixed = repaired_entity_labels(
                            key,
                            labels,
                            entity_types=self.definition.entity_types,
                        )
                        if not key or fixed is None:
                            continue
                        remove, add = canonical_label_changes(
                            labels,
                            fixed,
                            entity_types=self.definition.entity_types,
                        )
                        clauses = [*(f"REMOVE e:{label}" for label in remove)]
                        clauses.extend(f"SET e:{label}" for label in add)
                        if not clauses:
                            continue
                        result = session.run(
                            "MATCH (e:Entity {group_id: $gid, entity_key: $key}) "
                            + " ".join(clauses)
                            + " RETURN count(e) AS cnt",
                            gid=pot_id,
                            key=key,
                        )
                        rec = result.single()
                        result.consume()
                        repaired += int(rec["cnt"]) if rec is not None else 0
                    after = str(rows[-1].get("key") or "")
        finally:
            driver.close()
        return repaired


__all__ = ["Neo4jGraphBackend"]
