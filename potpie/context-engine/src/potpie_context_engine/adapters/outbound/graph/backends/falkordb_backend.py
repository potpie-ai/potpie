"""FalkorDB ``GraphBackend`` profile.

This backend wraps the existing FalkorDB reader/writer adapters behind the same
capability bundle used by Neo4j. Application services see only
``GraphBackend``; Falkor-specific graph handles, Lite/server mode, and Cypher
shim details stay in outbound adapters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping
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
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_analytics import (
    FalkorDBAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_inspection import (
    FalkorDBInspection,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    FalkorDBGraphProvider,
    FalkorDBGraphWriter,
    _records_from_result,
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
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.graph_mutations import ProvenanceContext
from potpie_context_core.lifecycle import DONE, FAILED, SetupPlan, StepResult
from potpie_context_core.ports.claim_query import ClaimQueryPort
from potpie_context_core.ports.graph.backend import BackendCapabilities
from potpie_context_core.ports.graph.mutation import BackendReadiness
from potpie_context_core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_core.reconciliation import MutationBatch, MutationResult
from potpie_context_core.reconciliation_config import ReconciliationConfig

_PROFILE = "falkordb"
_LITE_PROFILE = "falkordb_lite"


class _FalkorDBModeSettings:
    """Settings adapter that pins FalkorDB runtime mode for a backend profile."""

    def __init__(self, base: Any, mode: str) -> None:
        self._base = base
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def falkordb_mode(self) -> str:
        return self._mode


def _missing_driver_module(settings: Any) -> str | None:
    """The driver module this profile needs and does not have, if any.

    Readiness used to be answered from what was *wired* — ``writer.enabled`` is
    set during construction and stays true whether or not a handle can ever be
    opened. On a base ``potpie`` install, where the graph-native driver ships in
    the ``[local]`` extra, that produced the worst available ordering of two
    facts: ``potpie backend doctor`` said ``ready: true`` with every capability
    ``true``, and the very next read crashed. The diagnostic an operator runs
    first has to be the one that is right.

    A spec probe rather than an open: importability is exactly the question
    ("was the extra installed?"), and it costs nothing and starts no server.
    A driver that is present but broken is a different failure and still
    surfaces where it always did.
    """
    import importlib.util

    module = "falkordb" if settings.falkordb_mode() == "server" else "redislite"
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed installation
        found = False
    return None if found else module


def _run_sync(coro: Any) -> Any:
    """Drive an async writer call from a sync backend port."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "FalkorDBGraphBackend sync mutation cannot run inside an event loop; "
        "use the async door (mutation.apply_async)."
    )


@dataclass(slots=True)
class _FalkorDBMutation:
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
            detail="FalkorDB mutation receipts are not durable across processes",
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
        raise CapabilityNotImplemented(
            f"graph.{self.profile}.mutation.invalidate",
            detail=f"claim-key invalidation is not implemented for {self.profile} yet",
            recommended_next_action="use mutation.apply with InvalidationOp, or implement claim-key Cypher invalidation",
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return _run_sync(self.writer.reset_pot(pot_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        driver = _missing_driver_module(self.settings)
        ready = bool(getattr(self.writer, "enabled", False)) and driver is None
        if driver is not None:
            detail = (
                f"{self.profile} is selected but its driver ({driver!r}) is not "
                "installed — install it with `pip install 'potpie[local]'`, or "
                "use a managed host"
            )
        elif ready:
            detail = (
                f"{self.profile} claim_query + mutation + semantic + analytics + "
                "inspection wired; snapshot pending"
            )
        else:
            detail = (
                f"{self.profile} backend is not configured or context graph is disabled"
            )
        return BackendReadiness(
            profile=self.profile,
            ready=ready,
            detail=detail,
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
class FalkorDBGraphBackend:
    """FalkorDB-backed ``GraphBackend``."""

    settings: Any
    writer: GraphWriterPort | None = None
    graph_provider: FalkorDBGraphProvider | None = None
    embedder: Any = None
    profile_name: str = _PROFILE
    force_mode: str | None = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry,
        repr=False,
    )
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _FalkorDBMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        if self.force_mode is not None:
            self.settings = _FalkorDBModeSettings(self.settings, self.force_mode)
        provider = self.graph_provider or FalkorDBGraphProvider(self.settings)
        writer = self.writer or FalkorDBGraphWriter(
            self.settings,
            graph_provider=provider,
            embedder=self.embedder,
            definition=self.definition,
        )
        bind_writer = getattr(writer, "bind_definition", None)
        if callable(bind_writer):
            writer = bind_writer(self.definition)
        self.graph_provider = provider
        self.writer = writer
        self._claim_query = FalkorDBClaimQueryStore(
            self.settings, graph_provider=provider, embedder=self.embedder
        )
        self._mutation = _FalkorDBMutation(
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
        """Compatibility alias for old ingestion paths that seed via writer."""
        assert self.writer is not None
        return self.writer

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _FalkorDBMutation:
        return self._mutation

    @property
    def semantic(self) -> ClaimQuerySemanticSearch:
        return self._semantic

    @property
    def inspection(self) -> FalkorDBInspection:
        return FalkorDBInspection(
            self.settings,
            graph_provider=self.graph_provider,
            embedder=self.embedder,
            claim_query=self._claim_query,
        )

    @property
    def analytics(self) -> FalkorDBAnalytics:
        assert self.graph_provider is not None
        return FalkorDBAnalytics(
            graph_provider=self.graph_provider,
            fallback=ClaimQueryAnalytics(
                self._claim_query,
                entity_summary_repair=self._repair_entity_summaries,
                entity_label_repair=self._repair_entity_labels,
            ),
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

    def bind_definition(self, definition: GraphDefinition) -> FalkorDBGraphBackend:
        return replace(self, definition=definition)

    def provision(self, plan: SetupPlan) -> StepResult:
        if not self.enabled:
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=f"{self.profile_name} backend is not configured or context graph is disabled",
                metadata={"profile": self.profile_name},
            )
        try:
            ok = bool(_run_sync(self.graph_writer.ensure_indexes()))
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=str(exc),
                metadata={"profile": self.profile_name},
            )
        return StepResult(
            step="backend.provision",
            state=DONE if ok else FAILED,
            detail=(
                f"{self.profile_name} backend ready"
                if ok
                else f"{self.profile_name} index setup failed"
            ),
            metadata={"profile": self.profile_name},
        )

    def _repair_entity_summaries(self, pot_id: str) -> int:
        graph = self.graph_provider()
        rows = _records_from_result(
            graph.query(
                ENTITY_SUMMARY_SCAN_CYPHER,
                params={"gid": pot_id, "limit": ENTITY_SUMMARY_REPAIR_LIMIT},
            )
        )
        repaired = 0
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
            result = graph.query(
                ENTITY_SUMMARY_UPDATE_CYPHER,
                params={"gid": pot_id, "key": key, "props": fixed},
            )
            records = _records_from_result(result)
            if not records:
                repaired += 1
                continue
            repaired += int(records[0].get("cnt") or 0)
        return repaired

    def _repair_entity_labels(self, pot_id: str) -> int:
        graph = self.graph_provider()
        repaired = 0
        after = ""
        while True:
            rows = _records_from_result(
                graph.query(
                    ENTITY_LABEL_SCAN_CYPHER,
                    params={
                        "gid": pot_id,
                        "after": after,
                        "limit": ENTITY_LABEL_REPAIR_LIMIT,
                    },
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
                result = graph.query(
                    "MATCH (e:Entity {group_id: $gid, entity_key: $key}) "
                    + " ".join(clauses)
                    + " RETURN count(e) AS cnt",
                    params={"gid": pot_id, "key": key},
                )
                records = _records_from_result(result)
                repaired += int(records[0].get("cnt") or 0) if records else 1
            after = str(rows[-1].get("key") or "")
        return repaired


class FalkorDBLiteGraphBackend(FalkorDBGraphBackend):
    """FalkorDBLite-backed profile using the same Falkor adapter bundle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("profile_name", _LITE_PROFILE)
        kwargs.setdefault("force_mode", "lite")
        super().__init__(*args, **kwargs)


__all__ = ["FalkorDBGraphBackend", "FalkorDBLiteGraphBackend"]
