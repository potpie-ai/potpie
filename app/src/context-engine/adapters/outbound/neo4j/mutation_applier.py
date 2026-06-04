"""Bridge ``GraphMutationApplierPort`` to ``StructuralGraphPort`` generic methods."""

from __future__ import annotations

from adapters.outbound.neo4j.port import StructuralGraphPort
from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert, InvalidationOp, ProvenanceRef
from domain.ports.graph_mutation_applier import GraphMutationApplierPort


class StructuralGraphMutationApplier(GraphMutationApplierPort):
    def __init__(self, structural: StructuralGraphPort) -> None:
        self._structural = structural

    def apply_entity_upserts(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        return self._structural.upsert_entities(pot_id, items, provenance)

    def apply_edge_upserts(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        return self._structural.upsert_edges(pot_id, items, provenance)

    def apply_edge_deletes(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int:
        return self._structural.delete_edges(pot_id, items, provenance)

    def apply_invalidations(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int:
        return self._structural.apply_invalidations(pot_id, items, provenance)
