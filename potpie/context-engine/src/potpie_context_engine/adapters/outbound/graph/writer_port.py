"""Shared graph writer port.

Concrete graph stores implement this narrow async mutation adapter. It is
intentionally storage-adapter internal: application services depend on
``GraphBackend`` / ``GraphMutationPort`` instead.
"""

from __future__ import annotations

from typing import Any, Protocol

from potpie_context_engine.domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)


class GraphWriterPort(Protocol):
    """The deterministic graph mutation surface used by backend adapters."""

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
