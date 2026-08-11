"""Unit tests for the graph-explorer UI router helpers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie.cli import hosts
from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import (
    _caption,
    _node_type,
    _parse_scope,
    _slice_to_graph,
    build_ui_api_router,
)
from potpie_context_core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)

_TOKEN = "ui-router-test-token"  # noqa: S105 - non-secret test fixture


def _client(app: FastAPI) -> TestClient:
    """A client carrying the daemon token.

    ``/ui/api`` is authenticated; these tests are about routing and shaping, so
    they hold the credential the CLI holds rather than restating the gate (see
    tests/unit/test_ui_api_auth.py for that).
    """
    app.state.ui_auth = UiAuth(token=_TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {_TOKEN}"})


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    """Keep the developer's own managed login out of these tests.

    The router reads the host registry from the shared home — that is how the
    daemon learns about a managed endpoint at all — so without this a machine
    with a managed host configured would have every test here reach for it.
    """
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    yield
    hosts.reset_for_tests()


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

    response = _client(app).get("/api/pots")

    assert response.status_code == 200
    body = response.json()
    assert body["pots"][0]["counts"]["claims"] == 0
    assert body["pots"][1]["source_count"] == 1
    assert body["pots"][1]["counts"] == {"claims": 82, "entities": 46}


def test_daemon_app_mounts_ui_api_and_static(monkeypatch) -> None:
    from potpie.daemon import main as daemon_main

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
        log_file="/tmp/potpie-daemon.log",  # noqa: S108 - fixed non-secret test path
    )
    client = TestClient(app, headers={"Authorization": "Bearer test-token"})

    pots = client.get("/ui/api/pots")
    assert pots.status_code == 200
    assert pots.json()["pots"][0]["id"] == "p1"

    ui = client.get("/ui")
    assert ui.status_code == 200


# --- two hosts, one explorer ----------------------------------------------
#
# The daemon serves the UI for the managed host as well as its own, so the
# browser never has to hold a remote token. What is pinned below is that a
# request lands on the host it named — reading a pot id from one graph against
# the other is the failure worth engineering against.


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


class _Pots:
    def __init__(self, pots, *, raises: Exception | None = None) -> None:
        self._pots = pots
        self._raises = raises
        self.used: list[str] = []

    def list_pots(self):
        if self._raises is not None:
            raise self._raises
        return self._pots

    def active_pot(self):
        return next((p for p in self._pots if p.active), None)

    def use_pot(self, *, ref: str):
        self.used.append(ref)
        for pot in self._pots:
            if ref in (pot.pot_id, pot.name):
                return pot
        raise ValueError(f"no pot matching {ref!r}")

    def list_sources(self, *, pot_id):
        return []


class _Status:
    def __init__(self, origin: str) -> None:
        self.counts = {"claims": 1}
        self.backend_profile = origin
        self.backend_ready = True


class _Graph:
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def data_plane_status(self, pot_id):
        return _Status(self.origin)


class _FakeHost:
    def __init__(self, origin: str, pots: _Pots) -> None:
        self.pots = pots
        self.graph = _Graph(origin)


@pytest.fixture
def two_hosts(monkeypatch, tmp_path):
    """A local in-process host plus a configured, reachable managed one."""
    monkeypatch.setattr(
        hosts, "managed_endpoint", lambda: ("http://svc.example", "tok")
    )
    local = _FakeHost(
        "local",
        _Pots([_Pot("pot_l1", "default", active=True), _Pot("pot_l2", "notes")]),
    )
    managed = _FakeHost(
        "managed",
        _Pots([_Pot("pot_m1", "default"), _Pot("pot_m2", "api", active=True)]),
    )

    def _build(origin):
        if origin == hosts.LOCAL:
            raise AssertionError(
                "the router must use its in-process host for 'local', not an "
                "RPC client aimed back at this daemon"
            )
        return managed

    monkeypatch.setattr(hosts, "build_host", _build)

    app = FastAPI()
    app.include_router(build_ui_api_router(local))
    return _client(app), local, managed


def test_pots_api_merges_both_hosts_and_tags_origin(two_hosts) -> None:
    client, _, _ = two_hosts

    body = client.get("/api/pots").json()

    assert [(p["origin"], p["name"]) for p in body["pots"]] == [
        ("local", "default"),
        ("local", "notes"),
        ("managed", "default"),
        ("managed", "api"),
    ]
    # Both hosts have a pointer; only the active host's counts as *the* one.
    assert body["active_origin"] == "local"
    assert body["active"] == {"id": "pot_l1", "name": "default", "origin": "local"}


def test_pots_api_degrades_when_one_host_is_down(two_hosts, monkeypatch) -> None:
    """An empty selector reads as 'no pots'. The reachable host's pots are the
    useful part of the answer, so they still load and the failure is named."""
    client, _, _ = two_hosts
    monkeypatch.setattr(
        hosts,
        "build_host",
        lambda origin: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    body = client.get("/api/pots").json()

    assert {p["origin"] for p in body["pots"]} == {"local"}
    assert "connection refused" in body["unavailable"]["managed"]


def test_reads_route_to_the_host_they_name(two_hosts) -> None:
    client, _, _ = two_hosts

    local = client.get("/api/status", params={"host": "local", "pot": "pot_l1"}).json()
    managed = client.get(
        "/api/status", params={"host": "managed", "pot": "pot_m2"}
    ).json()

    assert (local["origin"], local["backend_profile"]) == ("local", "local")
    assert (managed["origin"], managed["backend_profile"]) == ("managed", "managed")


def test_a_pot_id_is_resolved_on_its_own_host_only(two_hosts) -> None:
    """`pot_m1` exists only on managed. Asking for it on local must 404 rather
    than silently falling through to the host that happens to have it."""
    client, _, _ = two_hosts

    response = client.get("/api/status", params={"host": "local", "pot": "pot_m1"})

    assert response.status_code == 404


def test_an_unknown_host_is_refused(two_hosts) -> None:
    client, _, _ = two_hosts

    response = client.get("/api/status", params={"host": "prod"})

    assert response.status_code == 400
    assert "prod" in response.json()["detail"]


def test_an_unreachable_host_is_unavailable_not_a_wrong_answer(
    two_hosts, monkeypatch
) -> None:
    client, _, _ = two_hosts
    monkeypatch.setattr(
        hosts,
        "build_host",
        lambda origin: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    response = client.get("/api/status", params={"host": "managed", "pot": "pot_m2"})

    # Targeting fails loud: falling back to local would answer from the wrong
    # graph, which is worse than not answering.
    assert response.status_code == 503


def test_selecting_a_managed_pot_moves_the_cli_pointer(two_hosts) -> None:
    """Picking a pot in the explorer is the same act as `potpie pot use`, so
    the terminal follows the selector instead of drifting from it."""
    client, local, managed = two_hosts

    body = client.post("/api/pots/use", json={"ref": "api", "host": "managed"}).json()

    assert body == {"id": "pot_m2", "name": "api", "origin": "managed", "active": True}
    assert managed.pots.used == ["api"] and local.pots.used == []
    assert hosts.persisted_origin() == "managed"


def test_a_qualified_ref_carries_its_own_host(two_hosts) -> None:
    client, _, managed = two_hosts

    body = client.post("/api/pots/use", json={"ref": "managed:api"}).json()

    assert body["origin"] == "managed"
    # The prefix is stripped before it reaches the host, which knows nothing
    # about origins and would not match a pot named 'managed:api'.
    assert managed.pots.used == ["api"]


def test_a_failed_switch_does_not_move_the_pointer(two_hosts) -> None:
    client, _, _ = two_hosts
    before = hosts.persisted_origin()

    response = client.post("/api/pots/use", json={"ref": "nope", "host": "managed"})

    assert response.status_code == 400
    assert hosts.persisted_origin() == before


def test_a_host_that_dies_mid_read_is_unavailable_not_a_crash(two_hosts) -> None:
    """Building the RPC client opens no connection, so a dead managed service
    surfaces on the first call, not in host resolution. Left unmapped it read
    to the SPA as 'the explorer is broken' rather than 'that host is down'."""
    from potpie_context_core.errors import ContextEngineDisabled

    client, _, managed = two_hosts
    managed.pots = _Pots([], raises=ContextEngineDisabled("connection refused"))

    response = client.get("/api/status", params={"host": "managed", "pot": "pot_m2"})

    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]


def test_remote_pot_counts_are_bounded(two_hosts, monkeypatch) -> None:
    """Each count costs two RPC round trips, so a few hundred remote pots turned
    the selector into seconds of blocking network. Past the budget the counts
    are omitted — never zeroed, which would read as 'this pot is empty'."""
    from potpie.daemon.http.ui import router as router_mod

    monkeypatch.setattr(router_mod, "_REMOTE_COUNT_LIMIT", 2)
    client, _, managed = two_hosts
    managed.pots = _Pots([_Pot(f"pot_m{i}", f"p{i}") for i in range(5)])

    body = client.get("/api/pots").json()
    remote = [p for p in body["pots"] if p["origin"] == "managed"]

    assert [("counts" in p) for p in remote] == [True, True, False, False, False]
    assert body["counts_complete"] is False


def test_counts_that_cannot_be_read_are_omitted_not_zeroed(two_hosts) -> None:
    """A backend that fails mid-listing used to report every one of that host's
    pots as holding nothing: `_counts` swallowed the error and returned `{}`,
    which is truthy in JS, so the selector rendered "0 claims" for a pot with
    three while `counts_complete` still claimed the numbers were all there."""

    class _Broken:
        def data_plane_status(self, pot_id):
            raise RuntimeError("graph backend is down")

    client, _, managed = two_hosts
    managed.graph = _Broken()

    body = client.get("/api/pots").json()
    remote = [p for p in body["pots"] if p["origin"] == "managed"]

    # The pots are still listed — you can open one, which is the point of the
    # selector; only the count is missing.
    assert [p["name"] for p in remote] == ["default", "api"]
    assert all("counts" not in p for p in remote)
    assert body["counts_complete"] is False
    assert "graph backend is down" in body["unavailable"]["managed counts"]


def test_the_in_process_host_is_never_budgeted(two_hosts, monkeypatch) -> None:
    """Local counts are in-process reads, not round trips — capping them would
    strip the selector's numbers for no gain."""
    from potpie.daemon.http.ui import router as router_mod

    monkeypatch.setattr(router_mod, "_REMOTE_COUNT_LIMIT", 1)
    client, local, _ = two_hosts
    local.pots = _Pots([_Pot(f"pot_l{i}", f"l{i}") for i in range(5)])

    body = client.get("/api/pots").json()
    local_rows = [p for p in body["pots"] if p["origin"] == "local"]

    assert all("counts" in p for p in local_rows)
