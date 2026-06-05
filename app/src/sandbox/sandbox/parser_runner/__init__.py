"""In-sandbox tree-sitter parser runner.

Invoked from the host as ``potpie-parse <repo_dir>``; emits the parsed
graph as NDJSON on stdout so the host can stream it into neo4j/qdrant
without holding the full payload in memory on either side.

Wire format (NDJSON, one JSON object per line, in order):

    {"kind": "header", "version": 1, "repo_dir": "..."}
    {"kind": "node", "id": "...", "node_type": "FILE", "file": "...", ...}
    {"kind": "node", ...}
    ...
    {"kind": "edge", "source_id": "...", "target_id": "...", "relationship_type": "...", ...}
    ...
    {"kind": "footer", "node_count": N, "edge_count": M, "elapsed_s": 12.34}

The host-side decoder lives at
``app/modules/intelligence/tools/sandbox/parser_wire.py`` and shares
the wire-version constant with this module — bump both when the
record schema changes.

Field set is sourced from the PyO3 ``GraphPayload`` exposed by the
``parsing_rs`` crate (``app/src/parsing/src/lib.rs``). This shim
mirrors that 1:1; ``build_node_attrs`` / ``build_edge_attrs`` on the
host (``parsing_repomap.py``) is the canonical consumer.
"""

from .runner import run

__all__ = ["run"]
