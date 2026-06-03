"""Code shared between Cypher-speaking backends (Neo4j, FalkorDB).

The two production backends share an internal write protocol: five typed
verbs over the canonical ``:RELATES_TO`` claim shape, driven by one
``apply_reconciliation_plan`` choreography. That code lives here, not in
``backends/neo4j/``, because both backends import from it — keeping it in
either backend's package would imply a sibling dependency.

External callers route through ``GraphMutationPort`` exposed by each
``GraphBackend.mutation`` (the canonical write contract). This package is
**not** a public seam — adapters within ``backends/*`` are the only consumers.
"""

from adapters.outbound.graph.backends._cypher_shared.apply_plan import (
    apply_reconciliation_plan,
)
from adapters.outbound.graph.backends._cypher_shared.writer import GraphWriterPort

__all__ = ["GraphWriterPort", "apply_reconciliation_plan"]
