"""Core parser-runner logic.

Kept import-light so ``python -m parser_runner`` cold-starts cheaply
inside the sandbox: only stdlib + ``parsing_rs`` (the Rust extension)
are touched here. The shim is intentionally dumb — it does not
walk the tree itself, doesn't filter ignores, doesn't talk to git.
All of that lives inside ``parsing_rs.extract_graph``; the runner
just serializes the result.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Iterable, TextIO

WIRE_VERSION = 1


def _node_record(node: Any) -> dict[str, Any]:
    """Mirror :class:`parsing_rs.NodePayload` as a JSON-friendly dict.

    Optional attributes (``class_name``, ``text``) are included only
    when set so the wire stays small for FILE nodes, which dominate
    the count and have no class name.
    """
    record: dict[str, Any] = {
        "kind": "node",
        "id": node.id,
        "node_type": node.node_type,
        "file": node.file,
        "line": node.line,
        "end_line": node.end_line,
        "name": node.name,
    }
    class_name = getattr(node, "class_name", None)
    if class_name is not None:
        record["class_name"] = class_name
    text = getattr(node, "text", None)
    if text is not None:
        record["text"] = text
    return record


def _edge_record(edge: Any) -> dict[str, Any]:
    """Mirror :class:`parsing_rs.RelationshipPayload`.

    Both ``relationship_type`` (canonical) and ``edge_type`` are tolerated
    on the host side; we only emit ``relationship_type`` because that's
    what the Rust payload actually carries.
    """
    record: dict[str, Any] = {
        "kind": "edge",
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relationship_type": edge.relationship_type,
    }
    ident = getattr(edge, "ident", None)
    if ident is not None:
        record["ident"] = ident
    ref_line = getattr(edge, "ref_line", None)
    if ref_line is not None:
        record["ref_line"] = ref_line
    end_ref_line = getattr(edge, "end_ref_line", None)
    if end_ref_line is not None:
        record["end_ref_line"] = end_ref_line
    return record


def _emit(out: TextIO, record: dict[str, Any]) -> None:
    out.write(json.dumps(record, ensure_ascii=False))
    out.write("\n")


def _iter_payload(payload: Any) -> Iterable[dict[str, Any]]:
    """Walk a :class:`parsing_rs.GraphPayload` and yield wire records.

    Tolerates older payloads that name the edge collection ``edges``
    (the host code does the same — see ``_reconstruct_graph_from_payload``).
    """
    for node in payload.nodes:
        yield _node_record(node)
    edges = getattr(payload, "edges", None)
    if edges is None:
        edges = payload.relationships
    for edge in edges:
        yield _edge_record(edge)


def run(
    repo_dir: str,
    *,
    out: TextIO | None = None,
    extract: Any | None = None,
) -> int:
    """Parse ``repo_dir`` and stream the graph as NDJSON to ``out``.

    ``extract`` is injected for tests; the production path imports
    :func:`parsing_rs.extract_graph` lazily so the runner module is
    importable in environments that don't have the wheel built (e.g.
    unit tests on the host).

    Returns the exit code (0 on success, non-zero on parser failure).
    """
    sink: TextIO = out if out is not None else sys.stdout
    if extract is None:
        # Imported lazily so unit tests on the host can stub `extract`.
        import parsing_rs  # type: ignore[import-not-found]

        extract_fn: Any = parsing_rs.extract_graph
    else:
        extract_fn = extract

    started = time.monotonic()
    _emit(sink, {"kind": "header", "version": WIRE_VERSION, "repo_dir": repo_dir})

    try:
        payload = extract_fn(repo_dir)
    except Exception as exc:
        # Surface the failure on stderr (caller propagates) and emit a
        # footer so the consumer can distinguish parser-internal failures
        # from premature pipe close.
        sys.stderr.write(f"parser_runner: extract_graph failed: {exc!r}\n")
        _emit(
            sink,
            {
                "kind": "footer",
                "node_count": 0,
                "edge_count": 0,
                "elapsed_s": time.monotonic() - started,
                "error": repr(exc),
            },
        )
        return 1

    node_count = 0
    edge_count = 0
    for record in _iter_payload(payload):
        if record["kind"] == "node":
            node_count += 1
        else:
            edge_count += 1
        _emit(sink, record)

    _emit(
        sink,
        {
            "kind": "footer",
            "node_count": node_count,
            "edge_count": edge_count,
            "elapsed_s": time.monotonic() - started,
        },
    )
    return 0


__all__ = ["run", "WIRE_VERSION"]
