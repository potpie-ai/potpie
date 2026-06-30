"""Wire format for the in-sandbox parser stream (host side).

The sandbox runner at ``app/src/sandbox/images/agent-sandbox/parser_runner/``
emits NDJSON on stdout; this module decodes it back into an
attribute-access payload that the existing host-side
``_reconstruct_graph_from_payload`` already understands. That keeps
the rest of the parsing pipeline (graph build, neo4j insert, qdrant
indexing) untouched.

``WIRE_VERSION`` here MUST match the constant in
``parser_runner/runner.py`` — the runner stamps it on every header and
the host rejects mismatched streams. Bump both when the field set
changes.

Stdlib-only on purpose: nothing here should drag in app code, so the
parsing pipeline can import it lazily without inflating cold-start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

WIRE_VERSION = 1


class WireFormatError(ValueError):
    """Raised when an NDJSON stream from the runner is malformed.

    Distinguishes parser-internal failures (carried as ``error`` on the
    footer record) from transport-level corruption like a truncated
    pipe or a mid-stream non-JSON line.
    """


@dataclass(slots=True)
class ParseArtifacts:
    """Parsed graph payload reconstructed from the runner's NDJSON.

    Mirrors :class:`parsing_rs.GraphPayload` closely enough that the
    existing host-side ``_reconstruct_graph_from_payload`` accepts it
    without modification: ``payload.nodes`` and ``payload.relationships``
    are iterables of attribute-access objects matching the PyO3 shapes.
    """

    nodes: list[SimpleNamespace] = field(default_factory=list)
    relationships: list[SimpleNamespace] = field(default_factory=list)
    repo_dir: str | None = None
    elapsed_s: float | None = None
    wire_version: int = WIRE_VERSION


def parse_stream(lines: Iterable[str]) -> ParseArtifacts:
    """Consume an NDJSON stream from ``potpie-parse`` and rebuild the payload.

    Tolerant of trailing whitespace and empty lines so callers can
    pass either ``stream.splitlines()`` or an iterator over a
    line-buffered stdout.
    """
    artifacts = ParseArtifacts()
    saw_header = False
    saw_footer = False
    error: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WireFormatError(
                f"non-JSON line in parser stream: {line[:120]!r}"
            ) from exc

        kind = record.get("kind")
        if kind == "header":
            if record.get("version") != WIRE_VERSION:
                raise WireFormatError(
                    f"unsupported wire version: {record.get('version')!r} "
                    f"(host expects {WIRE_VERSION})"
                )
            saw_header = True
            artifacts.repo_dir = record.get("repo_dir")
        elif kind == "node":
            artifacts.nodes.append(_node_from_record(record))
        elif kind == "edge":
            artifacts.relationships.append(_edge_from_record(record))
        elif kind == "footer":
            saw_footer = True
            artifacts.elapsed_s = record.get("elapsed_s")
            error = record.get("error")
            # Drain any trailing newlines after footer; ignore further records.
            break
        # Unknown kinds are tolerated for forward-compat: a newer runner
        # might emit additional record types alongside nodes/edges.

    if not saw_header:
        raise WireFormatError("parser stream missing header record")
    if not saw_footer:
        raise WireFormatError("parser stream truncated (no footer record)")
    if error is not None:
        raise WireFormatError(f"parser reported failure: {error}")

    return artifacts


def iter_node_records(payload: ParseArtifacts) -> Iterator[SimpleNamespace]:
    """Re-export so callers don't import :mod:`types` directly."""
    yield from payload.nodes


def iter_edge_records(payload: ParseArtifacts) -> Iterator[SimpleNamespace]:
    yield from payload.relationships


def _node_from_record(record: dict[str, Any]) -> SimpleNamespace:
    """Build the attribute-access object the host parser expects.

    Optional fields default to ``None`` so callers can do
    ``getattr(node, "class_name", None)`` and get sensible behavior.
    """
    return SimpleNamespace(
        id=record["id"],
        node_type=record["node_type"],
        file=record["file"],
        line=record["line"],
        end_line=record["end_line"],
        name=record["name"],
        class_name=record.get("class_name"),
        text=record.get("text"),
    )


def _edge_from_record(record: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        source_id=record["source_id"],
        target_id=record["target_id"],
        relationship_type=record["relationship_type"],
        ident=record.get("ident"),
        ref_line=record.get("ref_line"),
        end_ref_line=record.get("end_ref_line"),
    )


__all__ = [
    "ParseArtifacts",
    "WIRE_VERSION",
    "WireFormatError",
    "iter_edge_records",
    "iter_node_records",
    "parse_stream",
]
