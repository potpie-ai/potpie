"""Canonical graph backends, queries, mutation and journal adapters."""

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
    Neo4jClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort

__all__ = [
    "GraphWriterPort",
    "InMemoryClaimQueryStore",
    "Neo4jClaimQueryStore",
    "Neo4jGraphWriter",
]

# FalkorDB adapters are imported lazily (optional ``falkordb`` extra); they are
# intentionally NOT imported at module load so the default Neo4j install does
# not require the FalkorDB client. Import them directly from their modules
# (``potpie_context_engine.adapters.outbound.graph.falkordb_writer`` / ``...falkordb_reader``), or
# through the ``falkordb`` / ``falkordb_lite`` GraphBackend profiles.
