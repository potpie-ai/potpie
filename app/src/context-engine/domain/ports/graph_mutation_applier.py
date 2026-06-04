"""Structural mutations from validated plans (port)."""

from __future__ import annotations

from typing import Protocol

from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert, InvalidationOp, ProvenanceRef


class GraphMutationApplierPort(Protocol):
    def apply_entity_upserts(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        ...

    def apply_edge_upserts(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        ...

    def apply_edge_deletes(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int:
        ...

    def apply_invalidations(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int:
        ...
