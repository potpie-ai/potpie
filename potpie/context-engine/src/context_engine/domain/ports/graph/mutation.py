"""``GraphMutationPort`` — the canonical write path of a ``GraphBackend``.

Together with ``ClaimQueryPort`` this is the *source of truth*: every fact in
the graph enters through ``apply`` and is read back through claim query. The
other four capability ports are rebuildable projections off this store.

Records (``context_record``) and reconciled source events both lower to a
:class:`MutationBatch` and land here — there is one write door.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from context_engine.domain.graph_mutations import ProvenanceContext
from context_engine.domain.reconciliation import MutationBatch, MutationResult


@dataclass(frozen=True, slots=True)
class BackendReadiness:
    """Backend health/readiness, rolled into ``context_status`` backend
    readiness. Defined here (not in ``backend``) so capability ports can
    reference it without importing the bundle."""

    profile: str
    ready: bool
    detail: str | None = None
    capability_ready: Mapping[str, bool] = field(default_factory=dict)


class GraphMutationPort(Protocol):
    """Apply typed mutation plans and lifecycle operations to the claim store."""

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        """Apply a constrained mutation batch (entity/edge upserts, deletes,
        invalidations) against the canonical store for ``expected_pot_id``.

        Sync entry, for non-async callers (CLI, tests). Backends whose store is
        async (Neo4j) bridge with ``asyncio.run`` and therefore refuse to run
        inside a running event loop — async callers must use :meth:`apply_async`.
        """
        ...

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        """Async-native ``apply``. Preferred when called from inside an event
        loop (FastAPI handlers, agent tools, Celery workers): backends whose
        store is async (Neo4j) ``await`` their writer directly instead of
        bridging through ``asyncio.run`` — which raises inside a running loop and
        can cross-bind a connection pool to a dead loop. Sync backends just
        return ``apply(...)``. Mirror of ``ContextGraphPort.apply_plan_async``."""
        ...

    def invalidate(
        self,
        *,
        pot_id: str,
        claim_keys: Sequence[str],
        reason: str | None = None,
    ) -> int:
        """Soft-invalidate claims by key (stamp ``invalid_at``). Returns the
        number of claims invalidated."""
        ...

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        """Hard-reset all graph state for a pot. Returns a summary of what was
        removed."""
        ...

    def readiness(self, pot_id: str) -> BackendReadiness:
        """Cheap readiness probe for ``context_status`` / ``backend status``."""
        ...


__all__ = ["BackendReadiness", "GraphMutationPort"]
