"""FalkorDB ``GraphBackend`` — the lightweight Cypher profile.

Two profiles point at this package:

    falkor       → server-mode FalkorDB (needs FALKORDB_URL).
    falkor_lite  → embedded FalkorDBLite (no server, no Docker).

Both share the same writer/reader code; the mode is fixed at backend
construction by ``build_backend``. The internal Cypher write protocol +
apply-plan choreography live in ``backends/_cypher_shared`` (also used by
the Neo4j backend) — this package only holds the FalkorDB-specific pieces:
async-driver shim, mode-aware graph handle, unnamed index DDL, and
client-side batched ``reset_pot``.
"""

from adapters.outbound.graph.backends.falkor.backend import FalkorGraphBackend

__all__ = ["FalkorGraphBackend"]
