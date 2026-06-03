"""The graph layer.

One layer, two contracts: the writer side mutates the graph through the
internal ``GraphWriterPort`` (Neo4j + FalkorDB implementations, living in
``backends/_cypher_shared`` and each backend package), the reader side queries
canonical ``:RELATES_TO`` claims through ``ClaimQueryPort`` (Neo4j + FalkorDB +
in-memory). ``ContextGraphService`` is the application façade over a selected
``GraphBackend``.

Backends are chosen by profile via ``backends.build_backend`` —
``neo4j`` / ``falkor_lite`` / ``falkor`` / ``embedded`` / ``in_memory`` / the
registered stubs. Concrete writers are private to their backend package and are
**not** re-exported here; reach mutation through ``GraphBackend.mutation``.
"""

from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore

__all__ = [
    "ContextGraphService",
    "InMemoryClaimQueryStore",
    "Neo4jClaimQueryStore",
]
