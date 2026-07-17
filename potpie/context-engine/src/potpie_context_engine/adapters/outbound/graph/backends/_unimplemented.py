"""Capability stubs that fail closed with ``CapabilityNotImplemented``.

The skeleton wires every ``GraphBackend`` to all six capability ports — but a
profile that has not built a given projection yet uses one of these stubs
rather than leaving the slot ``None`` or raising a bare ``NotImplementedError``.
Inbound adapters catch ``CapabilityNotImplemented`` and render the structured
not-implemented contract (CLI exit 2/3, status ``not_implemented``).

Each stub stamps a dotted ``graph.<profile>.<capability>.<method>`` slot name so
the gap is attributable in logs/telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from potpie_context_engine.domain.errors import CapabilityNotImplemented
from potpie_context_engine.domain.graph_mutations import ProvenanceContext
from potpie_context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.domain.ports.graph.analytics import RepairReport
from potpie_context_engine.domain.ports.graph.inspection import GraphSlice
from potpie_context_engine.domain.ports.graph.mutation import BackendReadiness
from potpie_context_engine.domain.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.domain.reconciliation import MutationBatch, MutationResult


def _raise(profile: str, capability: str, method: str) -> Any:
    raise CapabilityNotImplemented(
        f"graph.{profile}.{capability}.{method}",
        detail=f"the '{profile}' backend has not implemented {capability}.{method} yet",
        recommended_next_action=f"use 'potpie backend use in_memory' or implement {capability} for '{profile}'",
    )


@dataclass(slots=True)
class UnimplementedClaimQuery:
    profile: str

    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        return _raise(self.profile, "claim_query", "find_claims")

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return _raise(self.profile, "claim_query", "entity_labels")


@dataclass(slots=True)
class UnimplementedMutation:
    profile: str

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        return _raise(self.profile, "mutation", "apply")

    def invalidate(
        self, *, pot_id: str, claim_keys: Sequence[str], reason: str | None = None
    ) -> int:
        return _raise(self.profile, "mutation", "invalidate")

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return _raise(self.profile, "mutation", "reset_pot")

    def readiness(self, pot_id: str) -> BackendReadiness:
        return BackendReadiness(
            profile=self.profile,
            ready=False,
            detail=f"'{self.profile}' backend mutation not implemented",
        )


@dataclass(slots=True)
class UnimplementedSemantic:
    profile: str

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        k: int = 10,
        filter_: ClaimQueryFilter | None = None,
    ) -> list[ClaimRow]:
        return _raise(self.profile, "semantic", "search")


@dataclass(slots=True)
class UnimplementedInspection:
    profile: str

    def neighborhood(
        self,
        *,
        pot_id: str,
        entity_key: str,
        depth: int = 1,
        direction: str = "both",
        predicates: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> GraphSlice:
        del direction, predicates, limit
        return _raise(self.profile, "inspection", "neighborhood")

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        return _raise(self.profile, "inspection", "path")

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return _raise(self.profile, "inspection", "labels")

    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        return _raise(self.profile, "inspection", "slice")


@dataclass(slots=True)
class UnimplementedAnalytics:
    profile: str

    def counts(self, pot_id: str) -> Mapping[str, int]:
        return _raise(self.profile, "analytics", "counts")

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        return _raise(self.profile, "analytics", "freshness")

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        return _raise(self.profile, "analytics", "quality")

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        return _raise(self.profile, "analytics", "repair")


@dataclass(slots=True)
class UnimplementedSnapshot:
    profile: str

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        return _raise(self.profile, "snapshot", "export")

    def import_(self, *, pot_id: str, source: str) -> SnapshotManifest:
        return _raise(self.profile, "snapshot", "import_")


__all__ = [
    "UnimplementedAnalytics",
    "UnimplementedClaimQuery",
    "UnimplementedInspection",
    "UnimplementedMutation",
    "UnimplementedSemantic",
    "UnimplementedSnapshot",
]
