"""FalkorDB ``GraphBackend`` profile.

This backend wraps the existing FalkorDB reader/writer adapters behind the same
capability bundle used by Neo4j. Application services see only
``GraphBackend``; Falkor-specific graph handles, Lite/server mode, and Cypher
shim details stay in outbound adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from potpie.context_engine.adapters.outbound.graph.apply_plan import apply_mutation_batch
from potpie.context_engine.adapters.outbound.graph.backends._unimplemented import (
    UnimplementedSnapshot,
)
from potpie.context_engine.adapters.outbound.graph.backends.claim_query_analytics import ClaimQueryAnalytics
from potpie.context_engine.adapters.outbound.graph.backends.claim_query_semantic import ClaimQuerySemanticSearch
from potpie.context_engine.adapters.outbound.graph.falkordb_inspection import FalkorDBInspection
from potpie.context_engine.adapters.outbound.graph.falkordb_reader import FalkorDBClaimQueryStore
from potpie.context_engine.adapters.outbound.graph.falkordb_writer import (
    FalkorDBGraphProvider,
    FalkorDBGraphWriter,
    _records_from_result,
)
from potpie.context_engine.adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_REPAIR_LIMIT,
    ENTITY_SUMMARY_SCAN_CYPHER,
    ENTITY_SUMMARY_UPDATE_CYPHER,
    repaired_entity_properties,
)
from potpie.context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie.context_engine.domain.errors import CapabilityNotImplemented
from potpie.context_engine.domain.graph_mutations import ProvenanceContext
from potpie.context_engine.domain.lifecycle import DONE, FAILED, SetupPlan, StepResult
from potpie.context_engine.domain.ports.claim_query import ClaimQueryPort
from potpie.context_engine.domain.ports.graph.backend import BackendCapabilities
from potpie.context_engine.domain.ports.graph.mutation import BackendReadiness
from potpie.context_engine.domain.reconciliation import MutationBatch, MutationResult

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

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        return await apply_mutation_batch(
            self.writer,
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
class FalkorDBGraphBackend:
    """FalkorDB-backed ``GraphBackend``."""

    settings: Any
    writer: GraphWriterPort | None = None
    graph_provider: FalkorDBGraphProvider | None = None
    embedder: Any = None
    profile_name: str = _PROFILE
    force_mode: str | None = None
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _FalkorDBMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        if self.force_mode is not None:
            self.settings = _FalkorDBModeSettings(self.settings, self.force_mode)
        provider = self.graph_provider or FalkorDBGraphProvider(self.settings)
        writer = self.writer or FalkorDBGraphWriter(
            self.settings, graph_provider=provider, embedder=self.embedder
        )
        self.graph_provider = provider
        self.writer = writer
        self._claim_query = FalkorDBClaimQueryStore(
            self.settings, graph_provider=provider, embedder=self.embedder
        )
        self._mutation = _FalkorDBMutation(
            self.settings, writer, profile=self.profile_name
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
        )

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        return ClaimQueryAnalytics(
            self._claim_query,
            entity_summary_repair=self._repair_entity_summaries,
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


class FalkorDBLiteGraphBackend(FalkorDBGraphBackend):
    """FalkorDBLite-backed profile using the same Falkor adapter bundle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("profile_name", _LITE_PROFILE)
        kwargs.setdefault("force_mode", "lite")
        super().__init__(*args, **kwargs)


__all__ = ["FalkorDBGraphBackend", "FalkorDBLiteGraphBackend"]
