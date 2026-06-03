"""The internal write protocol shared by Cypher-speaking backends.

Five typed-mutation verbs (``ensure_indexes``, ``upsert_entities`` /
``upsert_edges`` / ``delete_edges`` / ``invalidate`` / ``reset_pot``) over
the canonical Position-B ``:RELATES_TO`` edge layout. Both ``Neo4jGraphWriter``
and ``FalkorDBGraphWriter`` implement this protocol; ``apply_reconciliation_plan``
drives it. Outside the ``backends/*`` packages, the only mutation contract that
crosses package boundaries is :class:`GraphMutationPort` — never this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)


class GraphWriterPort(Protocol):
    """Internal Cypher write surface (shared between Neo4j + FalkorDB).

    Not a cross-package contract — every entry is shaped by Cypher's
    ``:RELATES_TO`` write semantics. Non-Cypher backends (in-memory,
    future Gremlin/Kuzu) implement :class:`GraphMutationPort` directly
    instead of this protocol.
    """

    @property
    def enabled(self) -> bool: ...

    async def ensure_indexes(self) -> bool:
        """Idempotently create entity / claim / vector indexes."""
        ...

    async def upsert_entities(
        self, pot_id: str, items: list[EntityUpsert], provenance: ProvenanceRef
    ) -> int: ...

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int: ...

    async def delete_edges(
        self, pot_id: str, items: list[EdgeDelete], provenance: ProvenanceRef
    ) -> int: ...

    async def invalidate(
        self, pot_id: str, items: list[InvalidationOp], provenance: ProvenanceRef
    ) -> int: ...

    async def reset_pot(self, pot_id: str) -> dict[str, Any]: ...


__all__ = ["GraphWriterPort"]
