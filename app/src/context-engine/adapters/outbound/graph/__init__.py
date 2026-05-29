"""The graph layer.

One layer, two contracts: the writer side mutates the graph through
``GraphWriterPort`` (one Neo4j implementation), the reader side queries
canonical ``:RELATES_TO`` claims through ``ClaimQueryPort`` (Neo4j +
in-memory implementations). ``ContextGraphService`` is the application
façade that composes both with the read orchestrator + optional LLM
answer / investigate paths.
"""

from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore
from adapters.outbound.graph.neo4j_writer import GraphWriterPort, Neo4jGraphWriter

__all__ = [
    "ContextGraphService",
    "GraphWriterPort",
    "InMemoryClaimQueryStore",
    "Neo4jClaimQueryStore",
    "Neo4jGraphWriter",
]

# FalkorDB adapters are imported lazily (optional ``falkordb`` extra); they are
# intentionally NOT imported at module load so the default Neo4j install does
# not require the FalkorDB client. Import them directly from their modules
# (``adapters.outbound.graph.falkordb_writer`` / ``...falkordb_reader``) or let
# ``build_container`` select them via ``GRAPH_DB_BACKEND=falkordb``.
