"""Unit tests for the local graph-explorer UI router helpers."""

from __future__ import annotations

from adapters.inbound.http.ui.router import (
    _caption,
    _node_type,
    _parse_scope,
    _slice_to_graph,
)
from domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice


def test_node_type_prefers_canonical_key_prefix_over_stray_label() -> None:
    # An Activity node that also accumulated a Dependency label must still
    # resolve to Activity (the key prefix is authoritative).
    assert (
        _node_type("activity:github:pr-848", ["Entity", "Dependency", "Activity"])
        == "Activity"
    )
    assert _node_type("repo:github.com/o/r", ["Entity", "Repository"]) == "Repository"


def test_node_type_falls_back_to_label_then_entity() -> None:
    assert _node_type("weird:thing", ["Entity", "Custom"]) == "Custom"
    assert _node_type("weird:thing", ["Entity"]) == "Entity"


def test_caption_prefers_summary_then_title_name_then_key_tail() -> None:
    assert (
        _caption("service:web", {"summary": "Web frontend service.", "name": "web"})
        == "Web frontend service."
    )
    assert _caption("activity:github:pr-1", {"title": "Add X"}) == "Add X"
    assert _caption("person:jane-doe", {"name": "Jane Doe"}) == "Jane Doe"
    assert _caption("repo:github.com/o/r", {}) == "github.com/o/r"


def test_caption_uses_description_for_old_nodes_without_summary() -> None:
    assert (
        _caption("service:web", {"description": "Web frontend service for browser clients."})
        == "Web frontend service for browser clients."
    )


def test_slice_to_graph_shape() -> None:
    sl = GraphSlice(
        pot_id="p",
        nodes=(
            GraphNode(key="repo:x", labels=("Entity", "Repository"), properties={"name": "x"}),
            GraphNode(key="person:y", labels=("Entity", "Person"), properties={}),
        ),
        edges=(GraphEdge(predicate="PERFORMED", from_key="person:y", to_key="repo:x"),),
        truncated=False,
    )
    g = _slice_to_graph(sl)
    assert [n["type"] for n in g["nodes"]] == ["Repository", "Person"]
    assert g["nodes"][0]["caption"] == "x"
    assert g["nodes"][0]["summary"] == "x"
    assert g["edges"][0] == {
        "id": "person:y|PERFORMED|repo:x",
        "source": "person:y",
        "target": "repo:x",
        "predicate": "PERFORMED",
    }
    assert g["truncated"] is False


def test_parse_scope() -> None:
    assert _parse_scope("repo:o/r,path:src/a.py") == {"repo": "o/r", "path": "src/a.py"}
    assert _parse_scope(None) == {}
    assert _parse_scope("bad,key:val") == {"key": "val"}
