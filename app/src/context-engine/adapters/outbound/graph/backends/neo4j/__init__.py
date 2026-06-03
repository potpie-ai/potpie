"""Neo4j ``GraphBackend`` — the shape-first production target.

Self-contained: the ``GraphWriterPort`` + ``Neo4jGraphWriter`` + the
``apply_reconciliation_plan`` choreography all live inside this package
because they are **Neo4j's private write shape**, not portable contracts.
External callers route through ``backend.mutation`` (the canonical
``GraphMutationPort``); nothing outside this package should import the
writer directly.
"""

from adapters.outbound.graph.backends.neo4j.backend import Neo4jGraphBackend

__all__ = ["Neo4jGraphBackend"]
