"""FalkorDB ``GraphBackend`` — lightweight Cypher backend for local dev.

Parallel structure to ``backends/neo4j/backend.py``: assemble the
canonical capability ports from the FalkorDB writer + reader, fail-close
the projections (semantic / inspection / snapshot) that aren't built out
yet, and compute analytics off the claim store.

Two profiles consume this class:

    falkor       — server mode; requires FALKORDB_URL.
    falkor_lite  — embedded FalkorDBLite over a local file. No server,
                   no Docker — just ``pip install context-engine[falkordb]``.

The writer and reader share **one** graph handle (via ``FalkorDBGraphProvider``)
because embedded FalkorDBLite cannot be opened twice on one db file.
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


def _run_sync(coro: Any) -> Any:
    """Drive a coroutine from a *sync* port entry (CLI/tests).

    Loop-aware: outside a running loop we run it with ``asyncio.run``; inside
    one we refuse rather than schedule onto the caller's loop. The FalkorDB
    writer's async work is ``asyncio.to_thread`` shims around sync calls, so
    nesting an ``asyncio.run`` from within a running loop would still
    deadlock — same bridge contract as Neo4j.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "FalkorGraphBackend sync mutation cannot run inside an event loop; "
        "use the async door (mutation.apply_async)."
    )


@dataclass(slots=True)
class _FalkorMutation:
    """``GraphMutationPort`` over ``FalkorDBGraphWriter`` + the shared apply choreography."""

    settings: Any
    mode: str
    graph_provider: Any = None
    writer: Any = None  # lazily created on first use

    def _get_writer(self) -> Any:
        if self.writer is None:
            from adapters.outbound.graph.backends.falkor.writer import (
                FalkorDBGraphWriter,
            )

            self.writer = FalkorDBGraphWriter(
                self.settings, mode=self.mode, graph_provider=self.graph_provider
            )
        return self.writer

    async def apply_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        from adapters.outbound.graph.backends._cypher_shared import (
            apply_reconciliation_plan,
        )

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

    def invalidate(self, *, pot_id: str, claim_keys: Any, reason: str | None = None) -> int:
        from domain.errors import CapabilityNotImplemented

        raise CapabilityNotImplemented(
            f"graph.falkor.mutation.invalidate",
            recommended_next_action="implement cypher invalidation by claim_key",
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return _run_sync(self._get_writer().reset_pot(pot_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        profile = f"falkor_{self.mode}" if self.mode == "lite" else "falkor"
        return BackendReadiness(
            profile=profile,
            ready=True,
            detail=(
                "falkordb claim_query + mutation + analytics wired; "
                "semantic/inspection/snapshot pending"
            ),
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
class FalkorGraphBackend:
    """FalkorDB-backed ``GraphBackend`` (mode-aware: ``lite`` or ``server``)."""

    settings: Any
    mode: str = "lite"
    _profile: str = field(init=False)
    _provider: Any = field(init=False)
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _FalkorMutation = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("lite", "server"):
            raise ValueError(
                f"unknown falkor mode '{self.mode}' (expected 'lite' or 'server')"
            )
        self._profile = "falkor_lite" if self.mode == "lite" else "falkor"
        # Lazy: only touch the falkordb driver when this profile is selected.
        from adapters.outbound.graph.backends.falkor.graph_handle import (
            FalkorDBGraphProvider,
        )
        from adapters.outbound.graph.backends.falkor.reader import (
            FalkorDBClaimQueryStore,
        )

        # One shared handle into writer + reader (load-bearing for Lite: two
        # handles on one db file would each spawn a redis-server).
        self._provider = FalkorDBGraphProvider(self.settings, mode=self.mode)
        self._claim_query = FalkorDBClaimQueryStore(
            self.settings, mode=self.mode, graph_provider=self._provider
        )
        self._mutation = _FalkorMutation(
            self.settings, mode=self.mode, graph_provider=self._provider
        )

    @property
    def enabled(self) -> bool:
        # Cheap config probe (no driver build); mirrors FalkorDBGraphWriter.enabled.
        is_enabled = getattr(self.settings, "is_enabled", None)
        if callable(is_enabled) and not is_enabled():
            return False
        if self.mode == "server":
            return bool(self.settings.falkordb_url())
        return True  # lite always enabled

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _FalkorMutation:
        return self._mutation

    @property
    def semantic(self) -> UnimplementedSemantic:
        return UnimplementedSemantic(self._profile)

    @property
    def inspection(self) -> UnimplementedInspection:
        return UnimplementedInspection(self._profile)

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        return ClaimQueryAnalytics(self._claim_query)

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(self._profile)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self._profile,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=False,
            inspection=False,
            snapshot=False,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        # FalkorDB is provisioned by either pip install (lite) or out-of-band
        # (server). Setup's job is to verify the handle builds and ensure
        # indexes — both idempotent and safe to re-run.
        from domain.lifecycle import DONE, FAILED, SKIPPED

        writer = self._mutation._get_writer()
        if not writer.enabled:
            return StepResult(
                step="backend.provision",
                state=SKIPPED,
                detail=(
                    "falkor disabled or unconfigured "
                    "(server mode needs FALKORDB_URL)"
                ),
                metadata={"profile": self._profile},
            )
        try:
            _run_sync(writer.ensure_indexes())
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=f"falkor unreachable or index setup failed: {exc}",
                metadata={"profile": self._profile},
            )
        return StepResult(
            step="backend.provision",
            state=DONE,
            detail="falkor reachable; canonical indexes ensured (best-effort)",
            metadata={"profile": self._profile},
        )


__all__ = ["FalkorGraphBackend"]
