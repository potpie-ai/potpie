"""LadybugDB ``GraphBackend`` profile (OSS embedded default)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any
import uuid

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionRegistry,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import (
    apply_mutation_batch,
)
from potpie_context_engine.adapters.outbound.graph.backends._unimplemented import (
    UnimplementedSnapshot,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_analytics import (
    LadybugAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_inspection import (
    LadybugInspection,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_reader import (
    LadybugClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (
    LadybugGraphProvider,
    LadybugGraphWriter,
)
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.ports.claim_query import ClaimQueryPort
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.mutation import (
    BackendReadiness,
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

_PROFILE = "ladybug"


def _run_sync(coro: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "LadybugGraphBackend sync mutation cannot run inside an event loop; "
        "use the async door (mutation.apply_async)."
    )


@dataclass(slots=True)
class _LadybugMutation:
    settings: Any
    writer: GraphWriterPort
    profile: str = _PROFILE
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry
    )

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
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
                self.writer,
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
            detail="Ladybug mutation receipts are not durable across processes",
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
        keys = tuple(claim_keys or ())
        invalidate_keys = getattr(self.writer, "invalidate_claim_keys", None)
        if not callable(invalidate_keys):
            from potpie_context_engine.core.errors import CapabilityNotImplemented

            raise CapabilityNotImplemented(
                f"graph.{self.profile}.mutation.invalidate",
                detail=(
                    f"claim-key invalidation is not implemented for {self.profile} yet"
                ),
                recommended_next_action=(
                    "use mutation.apply with InvalidationOp, or upgrade "
                    "potpie-context-engine to a build that implements "
                    "LadybugGraphWriter.invalidate_claim_keys"
                ),
            )
        return int(_run_sync(invalidate_keys(pot_id, keys, reason=reason)))

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return _run_sync(self.writer.reset_pot(pot_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        ready = bool(getattr(self.writer, "enabled", False))
        return BackendReadiness(
            profile=self.profile,
            ready=ready,
            detail=(
                f"{self.profile} claim_query + mutation + semantic + analytics + "
                "inspection wired; snapshot pending"
                if ready
                else f"{self.profile} backend is not configured or context graph is disabled"
            ),
            capability_ready={
                "mutation": ready,
                "claim_query": ready,
                "analytics": ready,
                "semantic": ready,
                "inspection": ready,
                "snapshot": False,
            },
        )


@dataclass(slots=True)
class LadybugGraphBackend:
    """Ladybug-backed ``GraphBackend`` (Claim-as-node + HNSW)."""

    settings: Any
    writer: GraphWriterPort | None = None
    graph_provider: LadybugGraphProvider | None = None
    embedder: Any = None
    profile_name: str = _PROFILE
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry,
        repr=False,
    )
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _LadybugMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        provider = self.graph_provider or LadybugGraphProvider(self.settings)
        writer = self.writer or LadybugGraphWriter(
            self.settings,
            conn_provider=provider,
            embedder=self.embedder,
            definition=self.definition,
        )
        bind_writer = getattr(writer, "bind_definition", None)
        if callable(bind_writer):
            writer = bind_writer(self.definition)
        self.graph_provider = provider
        self.writer = writer
        self._claim_query = LadybugClaimQueryStore(
            self.settings, conn_provider=provider, embedder=self.embedder
        )
        self._mutation = _LadybugMutation(
            self.settings,
            writer,
            profile=self.profile_name,
            definition=self.definition,
            execution_registry=self.execution_registry,
        )
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.writer, "enabled", False))

    @property
    def profile(self) -> str:
        return self.profile_name

    @property
    def graph_writer(self) -> GraphWriterPort:
        assert self.writer is not None
        return self.writer

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _LadybugMutation:
        return self._mutation

    @property
    def semantic(self) -> ClaimQuerySemanticSearch:
        return self._semantic

    @property
    def inspection(self) -> LadybugInspection:
        return LadybugInspection(
            self.settings,
            conn_provider=self.graph_provider,
            embedder=self.embedder,
            claim_query=self._claim_query,
        )

    @property
    def analytics(self) -> LadybugAnalytics:
        assert self.graph_provider is not None
        return LadybugAnalytics(
            conn_provider=self.graph_provider,
            fallback=ClaimQueryAnalytics(self._claim_query),
        )

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(self.profile_name)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self.profile_name,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=True,
            inspection=True,
            snapshot=False,
        )

    def bind_definition(self, definition: GraphDefinition) -> LadybugGraphBackend:
        return replace(self, definition=definition)

    def provision(self) -> BackendProvisionResult:
        if not self.enabled:
            return BackendProvisionResult(
                ok=False,
                detail=(
                    f"{self.profile_name} backend is not configured "
                    "or context graph is disabled"
                ),
                metadata={"profile": self.profile_name},
            )
        try:
            ok = bool(_run_sync(self.graph_writer.ensure_indexes()))
        except Exception as exc:  # noqa: BLE001
            return BackendProvisionResult(
                ok=False,
                detail=str(exc),
                metadata={"profile": self.profile_name},
            )
        return BackendProvisionResult(
            ok=ok,
            detail=(
                f"{self.profile_name} backend ready"
                if ok
                else f"{self.profile_name} schema setup failed"
            ),
            metadata={"profile": self.profile_name},
        )


__all__ = ["LadybugGraphBackend"]
