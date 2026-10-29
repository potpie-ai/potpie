"""The explorer against a managed host, served by the local daemon.

The daemon serves the explorer API for both hosts: ``?host=`` picks the graph,
the pot is resolved on that host only, and the managed host's key stays in the
daemon. The first half drives routing with in-memory sources; the second
drives the real :class:`ManagedExplorerSource` against the reference host,
including a host older than this client.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from potpie_context_engine.core.errors import ContextEngineDisabled
from potpie_context_engine.core.ports.graph.inspection import (
    GraphNode,
    GraphSlice,
)

from potpie.config import host_registry
from potpie.daemon.http.ui import build_ui_app
from potpie.daemon.http.ui.source import ManagedExplorerSource
from potpie.pots.contracts import PotInfo
from potpie.runtime import EngineOperation, remote
from tests.support.pot_directories import FakeDirectory
from tests.support.reference_host import ALICE_KEY, ReferenceHost

_TOKEN = "managed-explorer-daemon-token"  # noqa: S105 - non-secret test fixture
BASE = "https://your-host.example/graph"


def _slice(pot_id: str) -> GraphSlice:
    return GraphSlice(
        pot_id=pot_id,
        nodes=(GraphNode(key="service:web", labels=("Entity",), properties={}),),
        edges=(),
        truncated=False,
    )


class _Graph:
    """The local daemon's graph service for one fake pot store."""

    def __init__(self, profile: str = "embedded") -> None:
        self.profile = profile

    def data_plane_status(self, pot_id: str) -> Any:
        return SimpleNamespace(
            backend_profile=self.profile, backend_ready=True, counts={"claims": 3}
        )


class _Backend:
    class inspection:  # noqa: N801 - mirrors the backend attribute
        @staticmethod
        def slice(*, pot_id: str, filter_: Any) -> GraphSlice:
            return _slice(pot_id)

        @staticmethod
        def neighborhood(*, pot_id: str, entity_key: str, depth: int) -> GraphSlice:
            return _slice(pot_id)


class _Source:
    """An in-memory managed source."""

    origin = "managed"

    def __init__(self, pots: FakeDirectory) -> None:
        self.pots = pots
        self.counts_error: Exception | None = None
        self.read_error: Exception | None = None

    def engine_client(self, pot_id: str) -> Any:
        return None

    async def pot_counts(self, pots: list[Any]) -> dict[str, dict[str, Any]]:
        if self.counts_error is not None:
            raise self.counts_error
        return {p.pot_id: {"source_count": 1, "counts": {"claims": 7}} for p in pots}

    async def data_plane_status(self, pot_id: str) -> Any:
        if self.read_error is not None:
            raise self.read_error
        return SimpleNamespace(
            backend_profile="managed", backend_ready=True, counts={"claims": 7}
        )

    async def graph_slice(self, pot_id: str, *, include_invalidated: bool) -> Any:
        return _slice(pot_id)


@pytest.fixture
def two_hosts(monkeypatch):
    """A local daemon plus a configured, reachable managed host."""
    host_registry.reset_for_tests()
    monkeypatch.setattr(
        host_registry, "managed_endpoint", lambda: ("https://svc.example", "tok")
    )
    local = FakeDirectory(
        "local",
        [
            PotInfo(pot_id="pot_l1", name="default", active=True),
            PotInfo(pot_id="pot_l2", name="notes"),
        ],
    )
    managed = _Source(
        FakeDirectory(
            "managed",
            [
                PotInfo(pot_id="pot_m1", name="default"),
                PotInfo(pot_id="pot_m2", name="api", active=True),
            ],
        )
    )
    state: dict[str, Any] = {"managed": managed, "down": None}

    def _managed() -> Any:
        if state["down"] is not None:
            raise state["down"]
        return state["managed"]

    app = build_ui_app(
        pots=local,
        graph=_Graph(),
        backend=_Backend(),
        bearer_token=_TOKEN,
        managed_source=_managed,
    )
    client = TestClient(
        app,
        base_url="http://127.0.0.1:8765",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    yield SimpleNamespace(client=client, local=local, managed=managed, state=state)
    host_registry.reset_for_tests()


def test_pots_api_merges_both_hosts_and_tags_origin(two_hosts) -> None:
    body = two_hosts.client.get("/ui/api/pots").json()

    assert [(p["origin"], p["name"]) for p in body["pots"]] == [
        ("local", "default"),
        ("local", "notes"),
        ("managed", "default"),
        ("managed", "api"),
    ]
    assert body["active_origin"] == "local"
    assert body["active"] == {"id": "pot_l1", "name": "default", "origin": "local"}
    managed = [p for p in body["pots"] if p["origin"] == "managed"]
    # The host counts every pot in one answer: no budget, nothing omitted.
    assert all(p["counts"] == {"claims": 7} for p in managed)
    assert body["counts_complete"] is True


def test_pots_api_degrades_when_one_host_is_down(two_hosts) -> None:
    two_hosts.state["down"] = ContextEngineDisabled("connection refused")

    body = two_hosts.client.get("/ui/api/pots").json()

    assert {p["origin"] for p in body["pots"]} == {"local"}
    assert "connection refused" in body["unavailable"]["managed"]


def test_counts_that_cannot_be_read_are_omitted_not_zeroed(two_hosts) -> None:
    two_hosts.managed.counts_error = ContextEngineDisabled("counts unavailable")

    body = two_hosts.client.get("/ui/api/pots").json()

    managed = [p for p in body["pots"] if p["origin"] == "managed"]
    assert managed and all("counts" not in p for p in managed)
    assert body["counts_complete"] is False
    assert "counts unavailable" in body["unavailable"]["managed counts"]


def test_reads_route_to_the_host_they_name(two_hosts) -> None:
    client = two_hosts.client

    local = client.get("/ui/api/status", params={"host": "local"}).json()
    managed = client.get(
        "/ui/api/status", params={"host": "managed", "pot": "pot_m2"}
    ).json()

    assert (local["origin"], local["pot_id"]) == ("local", "pot_l1")
    assert (managed["origin"], managed["backend_profile"]) == ("managed", "managed")


def test_the_active_host_is_the_default(two_hosts) -> None:
    host_registry.set_persisted_origin("managed")

    body = two_hosts.client.get("/ui/api/status").json()

    assert (body["origin"], body["pot_id"]) == ("managed", "pot_m2")


def test_a_pot_id_is_resolved_on_its_own_host_only(two_hosts) -> None:
    response = two_hosts.client.get(
        "/ui/api/graph", params={"host": "local", "pot": "pot_m1"}
    )

    assert response.status_code == 404


def test_an_unknown_host_is_refused(two_hosts) -> None:
    response = two_hosts.client.get("/ui/api/status", params={"host": "elsewhere"})

    assert response.status_code == 400


def test_an_unusable_managed_host_is_unavailable_not_a_wrong_answer(
    two_hosts,
) -> None:
    two_hosts.state["down"] = ContextEngineDisabled("No managed host is configured.")

    response = two_hosts.client.get("/ui/api/status", params={"host": "managed"})

    assert response.status_code == 503
    assert "No managed host" in response.json()["detail"]


def test_a_host_that_dies_mid_read_is_unavailable_not_a_crash(two_hosts) -> None:
    two_hosts.managed.read_error = ContextEngineDisabled("connection reset")

    response = two_hosts.client.get(
        "/ui/api/status", params={"host": "managed", "pot": "pot_m2"}
    )

    assert response.status_code == 503


def test_selecting_a_managed_pot_moves_the_cli_pointer(two_hosts) -> None:
    body = two_hosts.client.post(
        "/ui/api/pots/use", json={"ref": "api", "host": "managed"}
    ).json()

    assert body == {"id": "pot_m2", "name": "api", "origin": "managed", "active": True}
    assert two_hosts.managed.pots.used == ["api"] and two_hosts.local.used == []
    assert host_registry.persisted_origin() == "managed"


def test_a_qualified_ref_carries_its_own_host(two_hosts) -> None:
    body = two_hosts.client.post("/ui/api/pots/use", json={"ref": "managed:api"}).json()

    assert body["origin"] == "managed"
    assert two_hosts.managed.pots.used == ["api"]


def test_a_failed_switch_does_not_move_the_pointer(two_hosts) -> None:
    response = two_hosts.client.post(
        "/ui/api/pots/use", json={"ref": "nope", "host": "managed"}
    )

    assert response.status_code == 404
    assert host_registry.persisted_origin() == "local"


# --- the real managed source against the reference host -------------------------


def _older_without_include_claims(document: dict[str, Any]) -> dict[str, Any]:
    """Play a host whose ``history`` predates ``include_claims``."""
    payload = document.get("payload")
    if document.get("operation") == "history" and isinstance(payload, dict):
        if "include_claims" in payload:
            payload["include_claims_unknown"] = payload.pop("include_claims")
    return document


def _rename_refused_field(document: Any) -> Any:
    if isinstance(document, dict):
        return {key: _rename_refused_field(value) for key, value in document.items()}
    if isinstance(document, list):
        return [_rename_refused_field(value) for value in document]
    if isinstance(document, str):
        return document.replace("include_claims_unknown", "include_claims")
    return document


@pytest.fixture
def reference_explorer(tmp_path: Path, monkeypatch):
    host_registry.reset_for_tests()
    monkeypatch.setattr(host_registry, "managed_endpoint", lambda: (BASE, ALICE_KEY))
    built: dict[str, Any] = {}

    def build(**host_kwargs: Any) -> Any:
        host = ReferenceHost.build(tmp_path / "host", **host_kwargs)
        host.state.active["alice"] = "pot_shop"
        seen: list[httpx.Request] = []
        connection = remote.connect(
            url=BASE,
            token=ALICE_KEY,
            http_transport=host.httpx_transport(
                seen=seen,
                rewrite_request=_older_without_include_claims,
                rewrite_response=_rename_refused_field,
            ),
        )
        app = build_ui_app(
            pots=FakeDirectory("local", []),
            graph=_Graph(),
            backend=_Backend(),
            bearer_token=_TOKEN,
            managed_source=lambda: ManagedExplorerSource(connection),
        )
        client = TestClient(
            app,
            base_url="http://127.0.0.1:8765",
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )
        built["host"] = host
        return SimpleNamespace(client=client, host=host, seen=seen)

    yield build
    host_registry.reset_for_tests()
    if "host" in built:
        remote._background.run(built["host"].close())


def _operations(seen: list[httpx.Request]) -> list[str]:
    return [json.loads(request.content)["operation"] for request in seen]


def test_the_whole_graph_is_the_graph_slice_operation(reference_explorer) -> None:
    explorer = reference_explorer()

    graph = explorer.client.get("/ui/api/graph", params={"host": "managed"})
    status = explorer.client.get("/ui/api/status", params={"host": "managed"})
    pots = explorer.client.get("/ui/api/pots").json()

    assert graph.status_code == 200, graph.text
    assert graph.json()["pot_id"] == "pot_shop"
    assert status.json()["origin"] == "managed"
    assert "graph_slice" in _operations(explorer.seen)
    managed = {p["id"]: p for p in pots["pots"] if p["origin"] == "managed"}
    assert managed["pot_shop"]["counts"] == {"claims": 0}
    assert managed["pot_shop"]["active"] is True
    # One host.pot_list with counts, not one call per pot.
    assert _operations(explorer.seen).count("host.pot_list") >= 1
    assert "host.source_list" not in _operations(explorer.seen)


def test_commit_history_on_an_older_host_is_a_structured_501(
    reference_explorer,
) -> None:
    explorer = reference_explorer(
        engine_operations=set(EngineOperation) - {EngineOperation.COMMITS}
    )

    response = explorer.client.get("/ui/api/commits", params={"host": "managed"})

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert detail["status"] == "host_outdated"
    assert "older Potpie" in detail["message"]
    assert detail["recommended_next_action"]
    assert detail["details"]["operation"] == "commits"


def test_plan_history_drops_an_option_an_older_host_lacks(reference_explorer) -> None:
    explorer = reference_explorer()

    response = explorer.client.get(
        "/ui/api/mutation-history", params={"host": "managed"}
    )

    assert response.status_code == 200, response.text
    sent = [
        json.loads(request.content)["payload"]
        for request in explorer.seen
        if json.loads(request.content)["operation"] == "history"
    ]
    # Compact requests: the default limit stays off the wire, and the retry
    # is exactly what a client that predates `include_claims` sends.
    assert sent == [{"include_claims": False}, {}]


def test_the_managed_key_never_reaches_the_browser(reference_explorer) -> None:
    explorer = reference_explorer()

    bodies = [
        explorer.client.get(path, params={"host": "managed"}).text
        for path in ("/ui/api/pots", "/ui/api/graph", "/ui/api/status")
    ]

    assert all(ALICE_KEY not in body for body in bodies)
    assert {request.headers["authorization"] for request in explorer.seen} == {
        f"Bearer {ALICE_KEY}"
    }


# --- potpie ui --pot managed:<ref> ----------------------------------------------


def test_ui_names_the_host_its_pot_was_resolved_on(monkeypatch, tmp_path) -> None:
    """The URL carries ``?host=`` beside the pot id, and the page is still
    served by the local daemon; nothing is persisted by opening it."""
    from typer import Typer
    from typer.testing import CliRunner

    from potpie.cli.commands import _common
    from potpie.cli.commands import ui as ui_cmd
    from potpie.daemon.discovery import write_daemon_credential
    from tests.support.pot_directories import two_hosts as wire_two_hosts

    host_registry.reset_for_tests()
    wire_two_hosts(monkeypatch)
    write_daemon_credential(tmp_path, "daemon-token-for-tests-0123456789abcdef")
    daemon = SimpleNamespace(
        home=tmp_path,
        ensure=lambda: None,
        status=lambda: {"ready": True, "url": "http://127.0.0.1:8765"},
    )
    monkeypatch.setattr(ui_cmd, "get_root_runtime", lambda: object())
    monkeypatch.setattr(ui_cmd, "get_daemon_service", lambda runtime: daemon)

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status_code = 200
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: _Response({"pots": [], "active_origin": "local"}),
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response({"code": "c1"}))
    _common.set_json(True)
    app = Typer()
    ui_cmd.register(app)

    try:
        result = CliRunner().invoke(app, ["--pot", "managed:api", "--no-open"])
    finally:
        _common.set_json(False)
        _common.set_pot_directory_factory(None)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "url": "http://127.0.0.1:8765/ui?host=managed&pot=pot_m2&k=c1",
        "pot_id": "pot_m2",
        "origin": "managed",
        "warning": None,
    }
    assert host_registry.persisted_origin() == "local"
