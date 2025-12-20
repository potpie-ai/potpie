"""Graph storage abstraction for Potpie.

See docs/graph_store_spec.md for full specification.
"""

from typing import Optional, Protocol, runtime_checkable

from potpie.types import CodeEdge, CodeNode


@runtime_checkable
class GraphStore(Protocol):
    """Graph storage protocol for code knowledge graphs.

    Implementations: NetworkXGraphStore (local), Neo4jGraphStore (server mode).
    """

    async def initialize(self) -> None:
        """Initialize graph storage (load from disk, connect to DB, etc.)."""
        ...

    async def close(self) -> None:
        """Close connections and persist if needed."""
        ...

    async def add_node(self, node: CodeNode) -> None:
        """Add a node to the graph. Updates if exists."""
        ...

    async def add_edge(self, edge: CodeEdge) -> None:
        """Add an edge between nodes."""
        ...

    async def get_node(self, node_id: str) -> Optional[CodeNode]:
        """Get a node by ID. Returns None if not found."""
        ...

    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out",
    ) -> list[CodeNode]:
        """Get neighboring nodes.

        Args:
            node_id: Starting node.
            edge_type: Filter by edge type (e.g., "CALLS", "IMPORTS").
            direction: "out" (outgoing), "in" (incoming), or "both".
        """
        ...

    async def query_nodes(
        self,
        node_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> list[CodeNode]:
        """Query nodes by type and/or name pattern."""
        ...

    async def clear(self) -> None:
        """Remove all nodes and edges."""
        ...


class NetworkXGraphStore:
    """NetworkX-based graph storage with JSON persistence.

    Stores graph in memory, persists to .potpie/graph.json.
    Implements GraphStore protocol.
    """

    def __init__(self, graph_path: str):
        """Initialize with path to graph JSON file.

        Args:
            graph_path: Path to persist graph data.
        """
        self.graph_path = graph_path
        self._graph = None

    async def initialize(self) -> None:
        """Load graph from disk or create new."""
        # TODO: Implement networkx graph initialization
        raise NotImplementedError("NetworkXGraphStore.initialize() not yet implemented")

    async def close(self) -> None:
        """Persist graph to disk."""
        # TODO: Implement graph persistence
        pass

    async def add_node(self, node: CodeNode) -> None:
        """Add a node to the graph."""
        # TODO: Implement add_node
        raise NotImplementedError("NetworkXGraphStore.add_node() not yet implemented")

    async def add_edge(self, edge: CodeEdge) -> None:
        """Add an edge to the graph."""
        # TODO: Implement add_edge
        raise NotImplementedError("NetworkXGraphStore.add_edge() not yet implemented")

    async def get_node(self, node_id: str) -> Optional[CodeNode]:
        """Get a node by ID."""
        # TODO: Implement get_node
        raise NotImplementedError("NetworkXGraphStore.get_node() not yet implemented")

    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out",
    ) -> list[CodeNode]:
        """Get neighboring nodes."""
        # TODO: Implement get_neighbors
        raise NotImplementedError(
            "NetworkXGraphStore.get_neighbors() not yet implemented"
        )

    async def query_nodes(
        self,
        node_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> list[CodeNode]:
        """Query nodes by type or name pattern."""
        # TODO: Implement query_nodes
        raise NotImplementedError(
            "NetworkXGraphStore.query_nodes() not yet implemented"
        )

    async def clear(self) -> None:
        """Clear all nodes and edges."""
        # TODO: Implement clear
        raise NotImplementedError("NetworkXGraphStore.clear() not yet implemented")
