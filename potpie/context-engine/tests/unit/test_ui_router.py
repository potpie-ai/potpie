"""Unit tests for the local graph-explorer UI router helpers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie_context_engine.adapters.inbound.http.ui.router import (
    _caption,
    _node_type,
    _parse_scope,
    _slice_to_graph,
    build_ui_api_router,
)
from potpie_context_engine.domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice


def test_node_type_prefers_canonical_key_prefix_over_stray_label() -> None:
    # An Activity node that also accumulated a Dependency label must still
    # resolve to Activity (the key prefix is authoritative).
    assert (
        _node_type("activity:github:pr-848", ["Entity", "Dependency", "Activity"])
        == "Activity"
    )
    assert _node_type("activity:github:pr-848", ["Entity", "Dependency"]) == "Activity"
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
        _caption(
            "service:web", {"description": "Web frontend service for browser clients."}
        )
        == "Web frontend service for browser clients."
    )


def test_slice_to_graph_shape() -> None:
    sl = GraphSlice(
        pot_id="p",
        nodes=(
            GraphNode(
                key="repo:x", labels=("Entity", "Repository"), properties={"name": "x"}
            ),
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


def test_pots_api_includes_counts_for_selector() -> None:
    class Pot:
        def __init__(self, pot_id, name, active=False):
            self.pot_id = pot_id
            self.name = name
            self.active = active

    class Pots:
        def __init__(self):
            self.p1 = Pot("p1", "empty", True)
            self.p2 = Pot("p2", "review")

        def list_pots(self):
            return [self.p1, self.p2]

        def active_pot(self):
            return self.p1

        def list_sources(self, *, pot_id):
            return [object()] if pot_id == "p2" else []

    class Status:
        def __init__(self, counts):
            self.counts = counts

    class Graph:
        def data_plane_status(self, pot_id):
            return Status(
                {"claims": 82, "entities": 46} if pot_id == "p2" else {"claims": 0}
            )

    class Host:
        pots = Pots()
        graph = Graph()

    app = FastAPI()
    app.include_router(build_ui_api_router(Host()))

    response = TestClient(app).get("/api/pots")

    assert response.status_code == 200
    body = response.json()
    assert body["pots"][0]["counts"]["claims"] == 0
    assert body["pots"][1]["source_count"] == 1
    assert body["pots"][1]["counts"] == {"claims": 82, "entities": 46}


def test_daemon_app_mounts_ui_api_and_static(monkeypatch) -> None:
    from potpie_context_engine.host import daemon_main

    class Pot:
        pot_id = "p1"
        name = "default"
        active = True

    class Pots:
        def list_pots(self):
            return [Pot()]

        def active_pot(self):
            return Pot()

        def list_sources(self, *, pot_id):
            return []

    class Status:
        counts = {"claims": 1}

    class Graph:
        def data_plane_status(self, pot_id):
            return Status()

    class Backend:
        profile = "in_memory"

    class Host:
        pots = Pots()
        graph = Graph()
        backend = Backend()

    monkeypatch.setattr(daemon_main, "build_host_shell", lambda: Host())

    app = daemon_main.create_app(
        token="test-token",
        base_url="http://127.0.0.1:1",
        pid=123,
        log_file="/tmp/potpie-daemon.log",
    )
    client = TestClient(app)

    pots = client.get("/ui/api/pots")
    assert pots.status_code == 200
    assert pots.json()["pots"][0]["id"] == "p1"

    ui = client.get("/ui")
    assert ui.status_code == 200
