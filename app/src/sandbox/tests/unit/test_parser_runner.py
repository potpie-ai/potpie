"""Tests for the in-sandbox parser runner shim and the host-side wire decoder.

The runner is at ``app/src/sandbox/sandbox/parser_runner/`` (lives inside
the ``potpie-sandbox`` package so it ships with the same install) and
the decoder at ``app/src/sandbox/sandbox/api/parser_wire.py``. These
are exercised together to verify the wire format round-trips — the
contract between the in-sandbox parser and the host pipeline.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Iterable

import pytest

from sandbox.api.parser_wire import (  # noqa: E402
    WIRE_VERSION,
    ParseArtifacts,
    WireFormatError,
    parse_stream,
)
from sandbox.parser_runner.runner import run as runner_run  # noqa: E402

pytestmark = pytest.mark.unit


def _node(**overrides) -> SimpleNamespace:
    """Build a NodePayload-shaped stub. Defaults match a typical FILE node."""
    base = dict(
        id="a.py",
        node_type="FILE",
        file="a.py",
        line=0,
        end_line=0,
        name="a.py",
        class_name=None,
        text="hello",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _edge(**overrides) -> SimpleNamespace:
    base = dict(
        source_id="a.py",
        target_id="Foo",
        relationship_type="CONTAINS",
        ident=None,
        ref_line=None,
        end_ref_line=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Payload:
    """Stand-in for :class:`parsing_rs.GraphPayload`."""

    def __init__(self, nodes: Iterable, edges: Iterable, *, edges_attr: str = "relationships") -> None:
        self.nodes = list(nodes)
        # Some older payloads name the attr `edges` instead of
        # `relationships`; the runner tolerates both, so we exercise
        # both via this kwarg.
        setattr(self, edges_attr, list(edges))


def _emit(payload: _Payload, repo: str = "/repo") -> str:
    out = io.StringIO()
    rc = runner_run(repo, out=out, extract=lambda _path: payload)
    assert rc == 0
    return out.getvalue()


# ---------------------------------------------------------------------------
# runner emit tests
# ---------------------------------------------------------------------------


def test_runner_emits_header_then_records_then_footer():
    payload = _Payload(nodes=[_node()], edges=[_edge()])
    output = _emit(payload).splitlines()

    kinds = [json.loads(line)["kind"] for line in output]
    assert kinds == ["header", "node", "edge", "footer"]

    header = json.loads(output[0])
    assert header == {"kind": "header", "version": WIRE_VERSION, "repo_dir": "/repo"}

    footer = json.loads(output[-1])
    assert footer["node_count"] == 1
    assert footer["edge_count"] == 1
    assert footer["elapsed_s"] >= 0
    assert "error" not in footer


def test_runner_omits_optional_fields_when_unset():
    """class_name / text / ident / ref_line absent ⇒ not in record.

    Keeps the wire small for FILE nodes (the dominant population) and
    matches the optionality the host-side ``build_node_attrs`` already
    encodes by checking ``is not None``.
    """
    payload = _Payload(
        nodes=[_node(class_name=None, text=None)],
        edges=[_edge(ident=None, ref_line=None, end_ref_line=None)],
    )
    output = _emit(payload).splitlines()

    node_record = json.loads(output[1])
    assert "class_name" not in node_record
    assert "text" not in node_record

    edge_record = json.loads(output[2])
    assert "ident" not in edge_record
    assert "ref_line" not in edge_record
    assert "end_ref_line" not in edge_record


def test_runner_includes_optional_fields_when_set():
    payload = _Payload(
        nodes=[_node(class_name="Foo", text="class Foo: pass")],
        edges=[_edge(ident="bar", ref_line=12, end_ref_line=15)],
    )
    output = _emit(payload).splitlines()

    node_record = json.loads(output[1])
    assert node_record["class_name"] == "Foo"
    assert node_record["text"] == "class Foo: pass"

    edge_record = json.loads(output[2])
    assert edge_record["ident"] == "bar"
    assert edge_record["ref_line"] == 12
    assert edge_record["end_ref_line"] == 15


def test_runner_tolerates_edges_attr_named_differently():
    """The PyO3 type names the collection ``relationships`` but older
    snapshots used ``edges``; the runner reads either."""
    payload = _Payload(nodes=[_node()], edges=[_edge()], edges_attr="edges")
    output = _emit(payload).splitlines()
    kinds = [json.loads(line)["kind"] for line in output]
    assert kinds == ["header", "node", "edge", "footer"]


def test_runner_returns_failure_when_extract_raises():
    """A parser-internal failure should mark the footer with ``error``
    and return a non-zero exit. The host distinguishes this from a
    pipe truncation by checking the footer."""
    out = io.StringIO()

    def _broken(_path):
        raise RuntimeError("syntax error in repo")

    rc = runner_run("/repo", out=out, extract=_broken)
    assert rc == 1

    lines = out.getvalue().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["header", "footer"]
    footer = json.loads(lines[-1])
    assert "syntax error" in footer["error"]


# ---------------------------------------------------------------------------
# parse_stream tests (host-side decoder)
# ---------------------------------------------------------------------------


def test_parse_stream_round_trip_basic():
    payload = _Payload(nodes=[_node(), _node(id="b.py", file="b.py")], edges=[_edge()])
    artifacts = parse_stream(_emit(payload).splitlines())
    assert isinstance(artifacts, ParseArtifacts)
    assert artifacts.repo_dir == "/repo"
    assert len(artifacts.nodes) == 2
    assert len(artifacts.relationships) == 1
    assert artifacts.elapsed_s is not None
    # The decoded objects are SimpleNamespace so attribute access works
    # with the existing host reconstructor without any glue code.
    assert artifacts.nodes[0].id == "a.py"
    assert artifacts.relationships[0].source_id == "a.py"


def test_parse_stream_rejects_missing_header():
    with pytest.raises(WireFormatError, match="missing header"):
        parse_stream([
            json.dumps({"kind": "node", "id": "x", "node_type": "FILE",
                        "file": "x", "line": 0, "end_line": 0, "name": "x"}),
            json.dumps({"kind": "footer", "node_count": 1, "edge_count": 0,
                        "elapsed_s": 0.1}),
        ])


def test_parse_stream_rejects_truncated_stream():
    """A stream with header + nodes but no footer is a transport error
    (potpie-parse pipe was killed mid-flight). Distinct from a
    parser-internal failure (which carries `error` on the footer)."""
    with pytest.raises(WireFormatError, match="truncated"):
        parse_stream([
            json.dumps({"kind": "header", "version": WIRE_VERSION, "repo_dir": "/r"}),
            json.dumps({"kind": "node", "id": "x", "node_type": "FILE",
                        "file": "x", "line": 0, "end_line": 0, "name": "x"}),
        ])


def test_parse_stream_propagates_parser_error_in_footer():
    """When the runner emits a footer with `error`, the host should
    surface that as WireFormatError instead of returning empty artifacts."""
    with pytest.raises(WireFormatError, match="syntax"):
        parse_stream([
            json.dumps({"kind": "header", "version": WIRE_VERSION, "repo_dir": "/r"}),
            json.dumps({"kind": "footer", "node_count": 0, "edge_count": 0,
                        "elapsed_s": 0.1, "error": "syntax error"}),
        ])


def test_parse_stream_rejects_wire_version_mismatch():
    with pytest.raises(WireFormatError, match="wire version"):
        parse_stream([
            json.dumps({"kind": "header", "version": 999, "repo_dir": "/r"}),
            json.dumps({"kind": "footer", "node_count": 0, "edge_count": 0,
                        "elapsed_s": 0}),
        ])


def test_parse_stream_rejects_non_json_line():
    with pytest.raises(WireFormatError, match="non-JSON"):
        parse_stream([
            json.dumps({"kind": "header", "version": WIRE_VERSION, "repo_dir": "/r"}),
            "this is not json at all",
            json.dumps({"kind": "footer", "node_count": 0, "edge_count": 0,
                        "elapsed_s": 0}),
        ])


def test_parse_stream_tolerates_unknown_kinds():
    """Forward-compat: a newer runner might emit additional record types
    alongside nodes/edges; the host should ignore them rather than
    fail (and ignore them silently rather than warn — they're inert)."""
    artifacts = parse_stream([
        json.dumps({"kind": "header", "version": WIRE_VERSION, "repo_dir": "/r"}),
        json.dumps({"kind": "diagnostic", "msg": "from a future runner"}),
        json.dumps({"kind": "footer", "node_count": 0, "edge_count": 0,
                    "elapsed_s": 0}),
    ])
    assert artifacts.nodes == []
    assert artifacts.relationships == []


def test_parse_stream_skips_blank_lines():
    """Trailing newlines from a subprocess pipe shouldn't bother us."""
    raw = "\n".join([
        json.dumps({"kind": "header", "version": WIRE_VERSION, "repo_dir": "/r"}),
        "",
        json.dumps({"kind": "footer", "node_count": 0, "edge_count": 0,
                    "elapsed_s": 0}),
        "",
    ])
    artifacts = parse_stream(raw.splitlines())
    assert artifacts.repo_dir == "/r"
