"""The ``GraphBackend`` capability bundle.

A ``GraphBackend`` is the swappable storage substrate for one pot's graph.
It is intentionally a *bundle of six narrow capability ports* rather than one
fat interface, so a backend can implement the canonical claim store (the
source of truth) while leaving rebuildable projections — semantic, inspection,
analytics, snapshot — as ``CapabilityNotImplemented`` until built.

    canonical (source of truth):  mutation + claim_query
    rebuildable projections:      semantic + inspection + analytics + snapshot

Profiles (``profile`` property) select the concrete backend:

    in_memory   real, used for tests + conformance
    falkordb_lite OSS local default on Python >=3.12
    embedded    JSON-persisted local fallback
    falkordb    external FalkorDB profile
    neo4j       shape-first production target (delegates to existing Neo4j code)
    hosted      managed profile (TODO)

Adding a backend means implementing these six ports behind a new profile —
never widening the public agent contract. See ``adapters/outbound/graph/backends/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from potpie.context_engine.domain.lifecycle import SetupPlan, StepResult
from potpie.context_engine.domain.ports.claim_query import ClaimQueryPort
from potpie.context_engine.domain.ports.graph.analytics import GraphAnalyticsPort
from potpie.context_engine.domain.ports.graph.inspection import GraphInspectionPort
from potpie.context_engine.domain.ports.graph.mutation import BackendReadiness, GraphMutationPort
from potpie.context_engine.domain.ports.graph.semantic import SemanticSearchPort
from potpie.context_engine.domain.ports.graph.snapshot import GraphSnapshotPort


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Static declaration of which capability ports a backend really
    implements vs. raises ``CapabilityNotImplemented``.

    The CLI ``backend status`` / ``backend doctor`` commands read this so a
    user can see, per profile, what is real today without triggering errors.
    """

    profile: str
    mutation: bool = False
    claim_query: bool = False
    semantic: bool = False
    inspection: bool = False
    analytics: bool = False
    snapshot: bool = False

    def implemented(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "mutation",
                "claim_query",
                "semantic",
                "inspection",
                "analytics",
                "snapshot",
            )
            if getattr(self, name)
        )


@runtime_checkable
class GraphBackend(Protocol):
    """Swappable graph capability bundle for one storage profile.

    The bundle composes the six capability ports. Application services depend
    on this — never on a concrete store — so swapping ``in_memory`` for
    ``neo4j`` is a wiring change, not a service rewrite.
    """

    @property
    def profile(self) -> str: ...

    @property
    def mutation(self) -> GraphMutationPort: ...

    @property
    def claim_query(self) -> ClaimQueryPort: ...

    @property
    def semantic(self) -> SemanticSearchPort: ...

    @property
    def inspection(self) -> GraphInspectionPort: ...

    @property
    def analytics(self) -> GraphAnalyticsPort: ...

    @property
    def snapshot(self) -> GraphSnapshotPort: ...

    def capabilities(self) -> BackendCapabilities: ...

    def provision(self, plan: SetupPlan) -> StepResult:
        """Stand up this backend's own store, idempotently (the setup seam).

        A backend self-provisions: ``embedded`` ensures its local file, a
        ``postgres`` profile creates the DB + enables pgvector + runs DDL, a
        ``neo4j`` profile pulls its container and creates indexes. Called by the
        setup orchestrator as a hard step; raises ``CapabilityNotImplemented``
        until a profile builds it.
        """
        ...


__all__ = ["BackendCapabilities", "BackendReadiness", "GraphBackend"]
