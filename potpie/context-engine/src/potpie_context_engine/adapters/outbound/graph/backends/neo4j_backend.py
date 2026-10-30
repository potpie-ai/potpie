"""Neo4j ``GraphBackend`` — the shape-first production target.

This is the canonical-store profile for the migration. The backend assembles
it from capability adapters and delegates the two *source-of-truth* ports to the
existing, battle-tested Neo4j code; semantic search is backed by Neo4j's native
relationship vector index, while inspection/snapshot remain fail-closed stubs
until they are built out.

    claim_query  -> Neo4jClaimQueryStore           (existing, real)
    mutation     -> atomic batch + durable receipt and pot revision
    analytics    -> ClaimQueryAnalytics             (computed from claim_query, real)
    semantic     -> ClaimQuerySemanticSearch        (native vector via claim_query)
    inspection   -> CapabilityNotImplemented        # TODO(stage-N): cypher traversal
    snapshot     -> Neo4jSnapshot                    (transactional portable export/import)

Neo4j imports are lazy so the skeleton (and the in_memory profile) load without
the ``graph`` extra installed; a missing driver surfaces only when this profile
is actually selected.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_mutations import InvalidationOp, ProvenanceContext
from potpie_context_engine.core.mutation_execution import MutationCapabilities
from potpie_context_engine.core.ports.claim_query import ClaimQueryPort
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.mutation import (
    BackendReadiness,
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionRegistry,
)
from potpie_context_engine.adapters.outbound.graph.backends._unimplemented import (
    UnimplementedInspection,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

_PROFILE = "neo4j"


def _run_sync(coro: Any) -> Any:
    """Drive a coroutine from a *sync* port entry (CLI/tests).

    Loop-aware: outside a running loop we run it with ``asyncio.run``; inside one
    we refuse rather than bind the writer's async connection pool to a throwaway
    loop and corrupt it — async callers must use the ``*_async`` door. Mirrors
    the canonical GraphService mutation surface.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "Neo4jGraphBackend sync mutation cannot run inside an event loop; "
        "use the corresponding async method."
    )


@dataclass(slots=True)
class _Neo4jMutation:
    """``GraphMutationPort`` over ``Neo4jGraphWriter`` + ``apply_mutation_batch``.

    ``apply_async`` is the native door (it ``await``s the async writer directly);
    ``apply`` is a loop-aware sync bridge for CLI/tests. Each operation owns and
    closes its async driver on the caller's loop; transaction-bound writers
    borrow that operation's transaction and never create or close a driver.
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

    @property
    def atomic_mutations_supported(self) -> bool:
        return callable(getattr(self._get_writer(), "in_transaction", None))

    @property
    def capabilities(self) -> MutationCapabilities:
        native = self.atomic_mutations_supported
        return MutationCapabilities(
            atomic_writes=native,
            revision_check=native,
            execution_lookup=native,
            durable_receipts=native,
        )

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
        writer = self._get_writer()
        if callable(getattr(writer, "in_transaction", None)):
            from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import (
                apply_atomic,
            )

            return await apply_atomic(
                writer,
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=context,
                definition=self.definition,
                reconciliation_config=reconciliation_config,
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

    async def lookup_execution_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        writer = self._get_writer()
        if callable(getattr(writer, "in_transaction", None)):
            from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import (
                lookup_execution,
            )

            return await lookup_execution(
                writer,
                plan,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
            )
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

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        return _run_sync(
            self.lookup_execution_async(
                plan, expected_pot_id=expected_pot_id, mutation_id=mutation_id
            )
        )

    async def current_version_async(self, pot_id: str) -> int:
        from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import (
            current_version,
        )

        if not self.atomic_mutations_supported:
            raise NotImplementedError(
                "injected Neo4j writer has no native revision check"
            )
        return await current_version(self._get_writer(), pot_id)

    def current_version(self, pot_id: str) -> int:
        return _run_sync(self.current_version_async(pot_id))

    async def compare_and_apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import (
            apply_atomic,
        )

        if not self.atomic_mutations_supported:
            raise NotImplementedError(
                "injected Neo4j writer has no native atomic execution"
            )
        context = replace(
            provenance_context or ProvenanceContext(),
            mutation_id=(provenance_context.mutation_id if provenance_context else None)
            or uuid.uuid4().hex,
        )
        return await apply_atomic(
            self._get_writer(),
            plan,
            expected_pot_id=expected_pot_id,
            expected_version=expected_version,
            provenance_context=context,
            definition=self.definition,
            reconciliation_config=reconciliation_config,
        )

    def compare_and_apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        return _run_sync(
            self.compare_and_apply_async(
                plan,
                expected_pot_id=expected_pot_id,
                expected_version=expected_version,
                provenance_context=provenance_context,
                reconciliation_config=reconciliation_config,
            )
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
        keys = tuple(dict.fromkeys(claim_keys))
        if not keys:
            return 0
        result = self.apply(
            MutationBatch(
                invalidations=[
                    InvalidationOp(
                        target_entity_key=None,
                        target_edge=None,
                        target_claim_keys=keys,
                        reason=reason or "claim invalidated",
                    )
                ]
            ),
            expected_pot_id=pot_id,
        )
        return result.mutation_summary.invalidations_applied

    async def reset_pot_async(
        self, pot_id: str, *, mutation_id: str | None = None
    ) -> dict[str, Any]:
        if not self.atomic_mutations_supported:
            raise NotImplementedError("Neo4j reset requires native atomic mutations")
        if mutation_id is not None and (
            not isinstance(mutation_id, str) or not mutation_id
        ):
            raise ValueError("reset mutation_id must be a nonempty string")
        result = await self.apply_async(
            MutationBatch(reset_pot=True),
            expected_pot_id=pot_id,
            provenance_context=ProvenanceContext(mutation_id=mutation_id),
        )
        return {
            "ok": result.ok,
            **result.administrative_summary,
            "mutation_id": result.mutation_id,
            "execution": result.execution.to_dict() if result.execution else None,
        }

    def reset_pot(
        self, pot_id: str, *, mutation_id: str | None = None
    ) -> dict[str, Any]:
        return _run_sync(self.reset_pot_async(pot_id, mutation_id=mutation_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        ready = bool(getattr(self._get_writer(), "enabled", False))
        atomic = ready and self.atomic_mutations_supported
        return BackendReadiness(
            profile=_PROFILE,
            ready=ready,
            detail="neo4j claim_query + mutation + semantic + analytics + snapshot wired; inspection pending",
            capability_ready={
                "mutation": ready,
                "claim_query": ready,
                "analytics": ready,
                "atomic_mutation": atomic,
                "durable_mutation_receipts": atomic,
                "semantic": ready,
                "inspection": False,
                "snapshot": atomic,
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
        self.writer = self._mutation._get_writer()
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)

    @property
    def enabled(self) -> bool:
        # Cheap config probe (no driver build): graph availability for policy and
        # the ingestion enabled flag. Mirrors Neo4jGraphWriter.enabled.
        is_enabled = getattr(self.settings, "is_enabled", None)
        return bool(is_enabled()) if callable(is_enabled) else True

    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def journal(self):
        from potpie_context_engine.adapters.outbound.graph.unavailable_journal import (
            UnavailableJournal,
        )

        return UnavailableJournal("Neo4j journal and rollback are deferred")

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
        from potpie_context_engine.adapters.outbound.graph.neo4j_admin import (
            Neo4jAnalytics,
        )

        return Neo4jAnalytics(self._claim_query, self._mutation, self.definition)

    @property
    def snapshot(self) -> Any:
        from potpie_context_engine.adapters.outbound.graph.neo4j_snapshot import (
            Neo4jSnapshot,
        )

        return Neo4jSnapshot(self.settings, mutation=self._mutation)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=True,
            inspection=False,
            snapshot=self._mutation.atomic_mutations_supported,
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
            ok = bool(_run_sync(self.writer.ensure_indexes()))
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


__all__ = ["Neo4jGraphBackend"]
