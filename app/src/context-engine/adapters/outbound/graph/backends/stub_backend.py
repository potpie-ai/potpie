"""``StubGraphBackend`` — a registered-but-unbuilt ``GraphBackend`` profile.

Profiles named in the docs (`postgres`/pgvector, `chroma`, the `hosted` managed
profile) need a real seam *now* so `backend list` shows them and selecting one
returns the structured not-implemented contract instead of crashing on an
unknown name. This backend composes the six fail-closed capability stubs and a
``provision`` that raises ``CapabilityNotImplemented`` — the profile owner swaps
in a real backend behind the same ``GraphBackend`` interface (see the embedded /
neo4j backends for the shape).
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.outbound.graph.backends._unimplemented import (
    UnimplementedAnalytics,
    UnimplementedClaimQuery,
    UnimplementedInspection,
    UnimplementedMutation,
    UnimplementedSemantic,
    UnimplementedSnapshot,
)
from domain.errors import CapabilityNotImplemented
from domain.lifecycle import SetupPlan, StepResult
from domain.ports.graph.backend import BackendCapabilities


@dataclass(slots=True)
class StubGraphBackend:
    """Fail-closed ``GraphBackend`` for a documented-but-unbuilt profile."""

    _profile: str

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def enabled(self) -> bool:
        # Not provisioned — nothing to enable yet. (Not part of the Protocol;
        # mirrors Neo4jGraphBackend.enabled for callers that probe it.)
        return False

    @property
    def mutation(self) -> UnimplementedMutation:
        return UnimplementedMutation(self._profile)

    @property
    def claim_query(self) -> UnimplementedClaimQuery:
        return UnimplementedClaimQuery(self._profile)

    @property
    def semantic(self) -> UnimplementedSemantic:
        return UnimplementedSemantic(self._profile)

    @property
    def inspection(self) -> UnimplementedInspection:
        return UnimplementedInspection(self._profile)

    @property
    def analytics(self) -> UnimplementedAnalytics:
        return UnimplementedAnalytics(self._profile)

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(self._profile)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(profile=self._profile)  # all six False

    def provision(self, plan: SetupPlan) -> StepResult:
        raise CapabilityNotImplemented(
            f"graph.{self._profile}.provision",
            detail=f"the '{self._profile}' backend profile is registered but not implemented",
            recommended_next_action=(
                f"use 'potpie setup --backend embedded', or implement the '{self._profile}' backend"
            ),
        )


__all__ = ["StubGraphBackend"]
