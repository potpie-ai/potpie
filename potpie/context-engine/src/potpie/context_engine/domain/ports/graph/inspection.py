"""``GraphInspectionPort`` — structural traversal projection of a ``GraphBackend``.

Backs ``potpie graph inspect`` and the graph explorer: neighbourhoods, paths,
label lookup, and bounded subgraph slices. A rebuildable projection over the
canonical claim edges; not the agent read path (that is the readers over
``ClaimQueryPort``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from potpie.context_engine.domain.ports.claim_query import ClaimQueryFilter


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One entity in an inspection slice."""

    key: str
    labels: tuple[str, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One :RELATES_TO edge in an inspection slice."""

    predicate: str
    from_key: str
    to_key: str
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphSlice:
    """A bounded subgraph returned by inspection queries."""

    pot_id: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    truncated: bool = False


class GraphInspectionPort(Protocol):
    """Structural traversal over the canonical graph."""

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
        """Return the subgraph within ``depth`` hops of ``entity_key``."""
        ...

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        """Return a shortest path (if any) between two entities."""
        ...

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        """Bulk label lookup for entity keys."""
        ...

    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        """Return the subgraph matching a claim filter (explorer/export view)."""
        ...


__all__ = ["GraphEdge", "GraphInspectionPort", "GraphNode", "GraphSlice"]
