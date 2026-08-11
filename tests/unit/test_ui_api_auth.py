"""The daemon's ``/ui`` surface must ask for a credential.

``/rpc`` has always required the daemon token. ``/ui/api`` sat on the same
loopback port with nothing in front of it, so any process on this machine could
read the whole project-memory graph — and one credential-free
``POST /ui/api/pots/use`` moved the active pot *and* the active host, with the
daemon spending its stored managed token on behalf of whoever asked.

What is pinned here: every JSON route refuses an anonymous caller, the mutation
refuses before it reaches the host, the browser gets in through a single-use
handoff instead of holding the token, and the page itself still loads without a
credential (it has to, in order to run the handoff).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from potpie.cli import hosts
from potpie.daemon import main as daemon_main
from potpie.daemon.http.ui import auth as ui_auth
from potpie_context_core.ports.graph.inspection import GraphNode, GraphSlice

TOKEN = "daemon-token-for-tests"  # noqa: S105 - non-secret test fixture
BEARER = {"Authorization": f"Bearer {TOKEN}"}

#: Every ``/ui/api`` route, with a request that would succeed were it let
#: through. Checked against the app's own route table below, so a route added
#: later without an entry here fails rather than quietly shipping open.
API_CALLS: dict[tuple[str, str], dict[str, Any]] = {
    ("GET", "/ui/api/pots"): {},
    ("POST", "/ui/api/pots/use"): {"json": {"ref": "default"}},
    ("POST", "/ui/api/handoff"): {},
    ("GET", "/ui/api/catalog"): {},
    ("GET", "/ui/api/status"): {},
    ("GET", "/ui/api/search"): {"params": {"q": "web"}},
    ("GET", "/ui/api/graph"): {},
    ("GET", "/ui/api/neighborhood"): {"params": {"key": "repo:x"}},
    ("GET", "/ui/api/read"): {"params": {"subgraph": "code", "view": "features"}},
}


class _Pot:
    pot_id = "pot_1"
    name = "default"
    active = True


class _Pots:
    def __init__(self) -> None:
        self.used: list[str] = []

    def list_pots(self) -> list[_Pot]:
        return [_Pot()]

    def active_pot(self) -> _Pot:
        return _Pot()

    def list_sources(self, *, pot_id: str) -> list[Any]:
        return []

    def use_pot(self, *, ref: str) -> _Pot:
        self.used.append(ref)
        return _Pot()


class _Status:
    counts = {"claims": 3}
    backend_profile = "in_memory"
    backend_ready = True


class _Envelope:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _Graph:
    def data_plane_status(self, pot_id: str) -> _Status:
        return _Status()

    def catalog(self, request: Any) -> _Envelope:
        return _Envelope({"subgraphs": []})

    def search_entities(self, request: Any) -> _Envelope:
        return _Envelope({"entities": []})

    def read(self, request: Any) -> _Envelope:
        return _Envelope({"view": "features"})


class _Inspection:
    def _slice(self) -> GraphSlice:
        return GraphSlice(
            pot_id="pot_1",
            nodes=(GraphNode(key="repo:x", labels=("Entity",), properties={}),),
            edges=(),
            truncated=False,
        )

    def slice(self, *, pot_id: str, filter_: Any) -> GraphSlice:
        return self._slice()

    def neighborhood(self, *, pot_id: str, entity_key: str, depth: int) -> GraphSlice:
        return self._slice()


class _Backend:
    profile = "in_memory"
    inspection = _Inspection()


class _Host:
    def __init__(self) -> None:
        self.pots = _Pots()
        self.graph = _Graph()
        self.backend = _Backend()


@pytest.fixture
def host(monkeypatch, tmp_path) -> _Host:
    """A fake in-process host, with the registry pointed at a temp home.

    Both halves matter: the registry is read from the shared home (that is how
    the daemon learns about a managed endpoint), and ``pots/use`` writes the
    active-host pointer into it.
    """
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)
    shell = _Host()
    monkeypatch.setattr(daemon_main, "build_host_shell", lambda: shell)
    yield shell
    hosts.reset_for_tests()


@pytest.fixture
def app(host: _Host):
    # No `with TestClient(...)` anywhere below: the lifespan writes this
    # machine's real pid/discovery files.
    return daemon_main.create_app(
        token=TOKEN,
        base_url="http://127.0.0.1:1",
        pid=123,
        log_file="/tmp/potpie-daemon-test.log",  # noqa: S108 - fixed test path
    )


@pytest.fixture
def anonymous(app) -> TestClient:
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def authorized(app) -> TestClient:
    return TestClient(app, headers=BEARER, follow_redirects=False)


def _api_routes(app) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/ui/api")
        for method in (route.methods or set()) - {"HEAD", "OPTIONS"}
    }


def test_the_case_list_covers_every_ui_api_route(app) -> None:
    """The gate is only worth as much as its coverage: a route added without a
    case here would never be checked for the credential."""
    assert _api_routes(app) == set(API_CALLS)


@pytest.mark.parametrize(("method", "path"), sorted(API_CALLS))
def test_no_ui_api_route_answers_an_anonymous_caller(
    anonymous: TestClient, method: str, path: str
) -> None:
    response = anonymous.request(method, path, **API_CALLS[(method, path)])

    assert response.status_code == 401, response.text
    # Loud enough to act on: the SPA renders `detail` verbatim.
    assert "potpie ui" in response.json()["detail"]


@pytest.mark.parametrize(("method", "path"), sorted(API_CALLS))
def test_every_ui_api_route_answers_the_daemon_token(
    authorized: TestClient, method: str, path: str
) -> None:
    response = authorized.request(method, path, **API_CALLS[(method, path)])

    assert response.status_code == 200, response.text


def test_a_wrong_token_is_not_a_credential(anonymous: TestClient) -> None:
    response = anonymous.get(
        "/ui/api/pots", headers={"Authorization": "Bearer not-the-token"}
    )

    assert response.status_code == 401


def test_the_pot_switch_is_refused_before_it_reaches_the_host(
    anonymous: TestClient, host: _Host, monkeypatch
) -> None:
    """The mutation is the sharp end: one anonymous POST moved the active pot
    and the active host, so it has to be refused *before* anything is written,
    not caught somewhere after the fact."""
    persisted: list[str] = []
    monkeypatch.setattr(hosts, "set_persisted_origin", persisted.append)

    response = anonymous.post("/ui/api/pots/use", json={"ref": "default"})

    assert response.status_code == 401
    assert host.pots.used == []
    assert persisted == []


def test_the_daemon_token_still_switches_pots(
    authorized: TestClient, host: _Host
) -> None:
    body = authorized.post("/ui/api/pots/use", json={"ref": "default"}).json()

    assert body == {
        "id": "pot_1",
        "name": "default",
        "origin": "local",
        "active": True,
    }
    assert host.pots.used == ["default"]


# --- browser handoff -------------------------------------------------------
#
# A navigation cannot carry a header, and the daemon token must not sit in the
# address bar or in history — so `potpie ui` spends the token on a single-use
# code, and the shell trades that for an HttpOnly cookie and redirects.


def _mint(authorized: TestClient) -> str:
    response = authorized.post("/ui/api/handoff")
    assert response.status_code == 200
    return response.json()["code"]


def test_a_session_cookie_cannot_mint_more_codes(
    anonymous: TestClient, authorized: TestClient
) -> None:
    """Otherwise a page could turn the cookie it cannot read into a code it
    can, and hand that to something outside the browser."""
    anonymous.get(f"/ui/?k={_mint(authorized)}")
    assert anonymous.get("/ui/api/pots").status_code == 200

    assert anonymous.post("/ui/api/handoff").status_code == 401


def test_a_handoff_code_becomes_a_cookie_and_leaves_the_url(
    anonymous: TestClient, authorized: TestClient
) -> None:
    code = _mint(authorized)

    response = anonymous.get(f"/ui/?host=local&pot=pot_1&k={code}")

    assert response.status_code == 303
    # The code is gone from where the browser would remember it; everything the
    # SPA reads off the URL survives.
    assert response.headers["location"] == "/ui/?host=local&pot=pot_1"
    cookie = response.headers["set-cookie"]
    assert ui_auth.SESSION_COOKIE in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/ui" in cookie


def test_the_handoff_cookie_authenticates_the_api(
    anonymous: TestClient, authorized: TestClient
) -> None:
    assert anonymous.get("/ui/api/pots").status_code == 401

    anonymous.get(f"/ui/?k={_mint(authorized)}")

    assert anonymous.get("/ui/api/pots").status_code == 200


def test_a_handoff_code_is_single_use(app, authorized: TestClient) -> None:
    """The link lives on in shell history and in the terminal scrollback; a
    reusable code there would be a standing credential."""
    code = _mint(authorized)
    TestClient(app, follow_redirects=False).get(f"/ui/?k={code}")

    second = TestClient(app, follow_redirects=False)
    replay = second.get(f"/ui/?k={code}")

    assert "set-cookie" not in replay.headers
    assert second.get("/ui/api/pots").status_code == 401


def test_an_unknown_code_grants_nothing(anonymous: TestClient) -> None:
    response = anonymous.get("/ui/?k=guessed-code")

    assert response.status_code == 303
    assert "set-cookie" not in response.headers
    assert anonymous.get("/ui/api/pots").status_code == 401


def test_an_expired_code_grants_nothing(
    anonymous: TestClient, authorized: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(ui_auth, "HANDOFF_TTL_SECONDS", 0.0)
    code = _mint(authorized)

    anonymous.get(f"/ui/?k={code}")

    assert anonymous.get("/ui/api/pots").status_code == 401


# --- cross-origin ----------------------------------------------------------


def test_a_cross_origin_switch_is_refused_even_with_a_session(
    anonymous: TestClient, authorized: TestClient, host: _Host
) -> None:
    """SameSite should already keep the cookie off another site's request; this
    is the second lock, because the cost of a miss is the CLI silently moving
    onto a different host."""
    anonymous.get(f"/ui/?k={_mint(authorized)}")

    response = anonymous.post(
        "/ui/api/pots/use",
        json={"ref": "default"},
        headers={"Origin": "http://evil.example"},
    )

    assert response.status_code == 403
    assert host.pots.used == []


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost"])
def test_the_pages_own_origin_is_not_cross_origin(
    anonymous: TestClient, authorized: TestClient, host: _Host, name: str
) -> None:
    """Both names reach the same loopback daemon, and the browser echoes back
    whichever the URL used. (The test client serves on the default port, so an
    origin without one is this daemon's own.)"""
    anonymous.get(f"/ui/?k={_mint(authorized)}")

    response = anonymous.post(
        "/ui/api/pots/use",
        json={"ref": "default"},
        headers={"Origin": f"http://{name}", "Referer": f"http://{name}/ui/"},
    )

    assert response.status_code == 200
    assert host.pots.used == ["default"]


@pytest.mark.parametrize(
    "origin",
    [
        # DNS rebinding: the name still resolves to loopback, so the request
        # arrives — and `Origin` and `Host` agree, because the page picked both.
        "http://rebound.example",
        # The same daemon's names, on a port it is not on: a different server.
        "http://127.0.0.1:1",
        "http://localhost:31337",
        "http://not-a-url",
    ],
)
def test_an_origin_this_daemon_cannot_be_at_is_refused(
    anonymous: TestClient, authorized: TestClient, host: _Host, origin: str
) -> None:
    """The expectation has to be stated, not read off the request. Deriving it
    from the request's own ``Host`` header made the comparison circular — the
    two halves agreed by construction and every origin passed."""
    anonymous.get(f"/ui/?k={_mint(authorized)}")

    response = anonymous.post(
        "/ui/api/pots/use",
        json={"ref": "default"},
        headers={"Origin": origin, "Host": urlsplit(origin).netloc or "testserver"},
    )

    assert response.status_code == 403, response.text
    assert host.pots.used == []


# --- the shell itself ------------------------------------------------------


@pytest.mark.parametrize("path", ["/ui", "/ui/"])
def test_the_spa_shell_loads_without_a_credential(
    anonymous: TestClient, path: str
) -> None:
    """It has to: the page is what runs the handoff. It ships no graph data —
    all of that comes from the authenticated API."""
    response = anonymous.get(path)

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
