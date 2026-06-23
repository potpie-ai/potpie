"""The graph adapter layer.

Backends expose ``ClaimQueryPort`` for reads and ``GraphMutationPort`` /
``GraphWriterPort`` for writes. ``DefaultGraphService`` is the canonical
application data plane; ``ContextGraphService`` is only a legacy DTO shim over
that service for managed callers that have not migrated yet.
"""

from context_engine.adapters.outbound.graph.context_graph_service import ContextGraphService
from context_engine.adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from context_engine.adapters.outbound.graph.neo4j_reader import Neo4jClaimQueryStore
from context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from context_engine.adapters.outbound.graph.writer_port import GraphWriterPort

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
# (``adapters.outbound.graph.falkordb_writer`` / ``...falkordb_reader``), or
# through the ``falkordb`` / ``falkordb_lite`` GraphBackend profiles.
