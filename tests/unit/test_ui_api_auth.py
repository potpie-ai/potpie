"""The daemon's ``/ui`` surface must ask for a credential.

The explorer API binds loopback, but every other process on this machine is on
loopback too, and so is any web page the user visits. Without a gate any of
them could read the whole project-memory graph, and one anonymous
``POST /ui/api/pots/use`` moved the pot the CLI works against.

What is pinned here: every JSON route refuses an anonymous caller, the mutation
refuses before it reaches the pot service, the browser gets in through a
single-use handoff instead of holding the token, a page under another origin or
another host name (DNS rebinding) is turned away, and the page itself still
loads without a credential (it has to, in order to run the handoff).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie.daemon.http.ui import auth as ui_auth
from potpie.daemon.http.ui import build_ui_api_router, build_ui_app
from potpie_context_engine.core.ports.graph.inspection import GraphNode, GraphSlice

pytestmark = pytest.mark.unit

TOKEN = "daemon-token-for-tests"  # noqa: S105 - non-secret test fixture
BEARER = {"Authorization": f"Bearer {TOKEN}"}
ORIGIN = "http://127.0.0.1:8765"

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
    ("POST", "/ui/api/telemetry/session"): {"json": {"had_graph": True}},
    ("GET", "/ui/api/commits"): {},
    ("GET", "/ui/api/commit"): {"params": {"commit_id": "c1"}},
    ("GET", "/ui/api/journal"): {},
    ("GET", "/ui/api/mutation-history"): {},
    ("POST", "/ui/api/rollback/preview"): {
        "json": {
            "pot": "pot_1",
            "target_commit_id": "c1",
            "expected_head": "c1",
            "mode": "revert",
        }
    },
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


class _Document:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _EngineClient:
    """Answers every commit-history operation the explorer sends."""

    def __init__(self, pot_id: str) -> None:
        self.pot_id = pot_id

    async def _ok(self, payload: dict[str, Any]) -> Any:
        from potpie_context_engine import Success

        return Success(_Document({"ok": True, **payload}))

    async def commits(self, request: Any) -> Any:
        return await self._ok({"headers": [], "coverage": {}})

    async def commit_show(self, request: Any) -> Any:
        return await self._ok({"changes": []})

    async def journal_status(self, request: Any) -> Any:
        return await self._ok({"state": None})

    async def history(self, request: Any) -> Any:
        return await self._ok({"entries": []})

    async def revert_preview(self, request: Any) -> Any:
        return await self._ok({"preview": {"preview_id": "p1"}})

    async def rollback_preview(self, request: Any) -> Any:
        return await self._ok({"preview": {"preview_id": "p1"}})


@pytest.fixture
def pots() -> _Pots:
    return _Pots()


def _app(pots: _Pots, *, token: str = TOKEN) -> FastAPI:
    return build_ui_app(
        pots=pots,
        graph=_Graph(),
        backend=_Backend(),
        bearer_token=token,
        engine_client=_EngineClient,
    )


@pytest.fixture
def app(pots: _Pots) -> FastAPI:
    return _app(pots)


@pytest.fixture
def anonymous(app: FastAPI) -> TestClient:
    return TestClient(app, base_url=ORIGIN, follow_redirects=False)


@pytest.fixture
def authorized(app: FastAPI) -> TestClient:
    return TestClient(app, base_url=ORIGIN, headers=BEARER, follow_redirects=False)


def _api_routes(app: FastAPI) -> set[tuple[str, str]]:
    # `app.openapi()` enumerates included routers even with the docs URL off.
    return {
        (method.upper(), path)
        for path, methods in app.openapi()["paths"].items()
        if path.startswith("/ui/api")
        for method in methods
        if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}
    }


def _mint(authorized: TestClient) -> str:
    response = authorized.post("/ui/api/handoff")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["expires_in"] == 120
    return body["code"]


def _mint_unchecked(authorized: TestClient) -> str:
    return authorized.post("/ui/api/handoff").json()["code"]


def _sign_in(client: TestClient, authorized: TestClient) -> None:
    client.get("/ui", params={"k": _mint(authorized)})


# --- every route asks ------------------------------------------------------


def test_the_case_list_covers_every_ui_api_route(app: FastAPI) -> None:
    """The gate is only worth as much as its coverage: a route added without a
    case here would never be checked for the credential."""
    assert _api_routes(app) == set(API_CALLS)


@pytest.mark.parametrize(("method", "path"), sorted(API_CALLS))
def test_no_ui_api_route_answers_an_anonymous_caller(
    anonymous: TestClient, method: str, path: str
) -> None:
    response = anonymous.request(method, path, **API_CALLS[(method, path)])

    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "Bearer"
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


def test_a_non_ascii_credential_is_a_mismatch_not_a_crash(
    anonymous: TestClient,
) -> None:
    response = anonymous.get(
        "/ui/api/pots", headers={"Authorization": "Bearer töken".encode()}
    )

    assert response.status_code == 401


def test_a_router_mounted_without_a_credential_serves_nothing(pots: _Pots) -> None:
    """Fail closed: an app assembled without the daemon token must not fall
    back to serving the graph to whoever asks."""
    bare = FastAPI()
    bare.include_router(
        build_ui_api_router(pots=pots, graph=_Graph(), backend=_Backend()),
        prefix="/ui",
    )

    response = TestClient(bare).get("/ui/api/pots", headers=BEARER)

    assert response.status_code == 401


def test_the_pot_switch_is_refused_before_it_reaches_the_pot_service(
    anonymous: TestClient, pots: _Pots
) -> None:
    """The mutation is the sharp end: one anonymous POST moved the active pot,
    so it has to be refused before anything is written."""
    response = anonymous.post("/ui/api/pots/use", json={"ref": "default"})

    assert response.status_code == 401
    assert pots.used == []


def test_the_daemon_token_still_switches_pots(
    authorized: TestClient, pots: _Pots
) -> None:
    body = authorized.post("/ui/api/pots/use", json={"ref": "default"}).json()

    assert body == {"id": "pot_1", "name": "default", "active": True}
    assert pots.used == ["default"]


def test_framework_schema_and_docs_pages_are_not_served(
    anonymous: TestClient,
) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert anonymous.get(path).status_code == 404, path


# --- browser handoff -------------------------------------------------------
#
# A navigation cannot carry a header, and the daemon token must not sit in the
# address bar or in history -- so `potpie ui` spends the token on a single-use
# code, and the shell trades that for an HttpOnly cookie and redirects.


def test_a_session_cookie_cannot_mint_more_codes(
    anonymous: TestClient, authorized: TestClient
) -> None:
    """Otherwise a page could turn the cookie it cannot read into a code it
    can, and hand that to something outside the browser."""
    _sign_in(anonymous, authorized)
    assert anonymous.get("/ui/api/pots").status_code == 200

    assert anonymous.post("/ui/api/handoff").status_code == 401


def test_a_handoff_code_becomes_a_cookie_and_leaves_the_url(
    anonymous: TestClient, authorized: TestClient
) -> None:
    code = _mint(authorized)

    response = anonymous.get(f"/ui?pot=pot_1&k={code}")

    assert response.status_code == 303
    # The code is gone from where the browser would remember it; everything the
    # SPA reads off the URL survives.
    assert response.headers["location"] == "/ui?pot=pot_1"
    assert response.headers["cache-control"] == "no-store"
    cookie = response.headers["set-cookie"]
    assert ui_auth.SESSION_COOKIE in cookie
    assert code not in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/ui" in cookie


def test_the_handoff_cookie_authenticates_the_api(
    anonymous: TestClient, authorized: TestClient
) -> None:
    assert anonymous.get("/ui/api/pots").status_code == 401

    _sign_in(anonymous, authorized)

    assert anonymous.get("/ui/api/pots").status_code == 200


def test_opening_another_daemon_preserves_both_browser_sessions(
    app: FastAPI, pots: _Pots
) -> None:
    """Cookies share a host across ports; distinct daemons must not overwrite
    each other's credentials when their explorers open in the same browser."""
    other_token = "other-daemon-token"  # noqa: S105 - non-secret test fixture
    other_app = _app(pots, token=other_token)
    first = TestClient(app, base_url="http://127.0.0.1:8765")
    second = TestClient(other_app, base_url="http://127.0.0.1:8766")

    first_code = first.post("/ui/api/handoff", headers=BEARER).json()["code"]
    first.get("/ui", params={"k": first_code})
    assert first.get("/ui/api/pots").status_code == 200

    # A browser sends the same cookie jar to both loopback ports. The first
    # daemon's session must not authenticate the second daemon.
    second.cookies.update(first.cookies)
    assert second.get("/ui/api/pots").status_code == 401
    second_code = second.post(
        "/ui/api/handoff", headers={"Authorization": f"Bearer {other_token}"}
    ).json()["code"]
    second.get("/ui", params={"k": second_code})
    assert second.get("/ui/api/pots").status_code == 200

    # Bring the browser's updated jar back to the original tab.
    first.cookies.update(second.cookies)
    assert first.get("/ui/api/pots").status_code == 200


def test_a_handoff_code_is_single_use(app: FastAPI, authorized: TestClient) -> None:
    """The link lives on in shell history and terminal scrollback; a reusable
    code there would be a standing credential."""
    code = _mint(authorized)
    TestClient(app, base_url=ORIGIN, follow_redirects=False).get(f"/ui?k={code}")

    second = TestClient(app, base_url=ORIGIN, follow_redirects=False)
    replay = second.get(f"/ui?k={code}")

    assert replay.status_code == 303
    assert "set-cookie" not in replay.headers
    assert second.get("/ui/api/pots").status_code == 401


def test_an_unknown_code_grants_nothing(anonymous: TestClient) -> None:
    response = anonymous.get("/ui/?k=guessed-code")

    assert response.status_code == 303
    assert response.headers["location"] == "/ui/"
    assert "set-cookie" not in response.headers
    assert anonymous.get("/ui/api/pots").status_code == 401


def test_an_expired_code_grants_nothing(
    anonymous: TestClient, authorized: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(ui_auth, "HANDOFF_TTL_SECONDS", 0.0)
    code = _mint_unchecked(authorized)

    anonymous.get(f"/ui?k={code}")

    assert anonymous.get("/ui/api/pots").status_code == 401


def test_an_expired_session_is_refused(
    anonymous: TestClient, authorized: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(ui_auth, "SESSION_TTL_SECONDS", 0.0)
    _sign_in(anonymous, authorized)

    assert anonymous.get("/ui/api/pots").status_code == 401


def test_a_new_daemon_boot_forgets_every_session(
    anonymous: TestClient, authorized: TestClient, pots: _Pots
) -> None:
    """Sessions live in the daemon's memory: a restarted daemon (new token,
    new app) must not honour a cookie the old one issued."""
    _sign_in(anonymous, authorized)
    restarted = TestClient(_app(pots, token="next-boot-token"), base_url=ORIGIN)
    restarted.cookies.update(anonymous.cookies)

    assert restarted.get("/ui/api/pots").status_code == 401


# --- cross-origin ----------------------------------------------------------


def test_a_cross_origin_switch_is_refused_even_with_a_session(
    anonymous: TestClient, authorized: TestClient, pots: _Pots
) -> None:
    """SameSite should already keep the cookie off another site's request; this
    is the second lock."""
    _sign_in(anonymous, authorized)

    response = anonymous.post(
        "/ui/api/pots/use",
        json={"ref": "default"},
        headers={"Origin": "http://evil.example"},
    )

    assert response.status_code == 403
    assert pots.used == []


def test_a_cross_origin_read_is_refused_even_with_a_session(
    anonymous: TestClient, authorized: TestClient
) -> None:
    _sign_in(anonymous, authorized)

    response = anonymous.get(
        "/ui/api/graph", headers={"Referer": "http://evil.example/page"}
    )

    assert response.status_code == 403


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost"])
def test_the_pages_own_origin_is_not_cross_origin(
    anonymous: TestClient, authorized: TestClient, pots: _Pots, name: str
) -> None:
    """Both names reach the same loopback daemon, and the browser echoes back
    whichever the URL used."""
    _sign_in(anonymous, authorized)

    response = anonymous.post(
        "/ui/api/pots/use",
        json={"ref": "default"},
        headers={"Origin": f"http://{name}:8765", "Referer": f"http://{name}:8765/ui"},
    )

    assert response.status_code == 200, response.text
    assert pots.used == ["default"]


def test_the_session_beacon_from_the_page_is_accepted(
    anonymous: TestClient, authorized: TestClient
) -> None:
    _sign_in(anonymous, authorized)

    response = anonymous.post(
        "/ui/api/telemetry/session",
        json={"had_graph": False},
        headers={"Origin": ORIGIN},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.parametrize(
    "origin",
    [
        # DNS rebinding: the name still resolves to loopback, so the request
        # arrives -- and the page picked its own `Origin`.
        "http://rebound.example:8765",
        # The same daemon's names, on a port it is not on: a different server.
        "http://127.0.0.1:1",
        "http://localhost:31337",
        "http://127.0.0.1:not-a-port",
        "null",
    ],
)
def test_an_origin_this_daemon_cannot_be_at_is_refused(
    anonymous: TestClient, authorized: TestClient, pots: _Pots, origin: str
) -> None:
    """The expectation is stated (loopback names, the port the socket is bound
    to), not read off the request's own ``Host`` header."""
    _sign_in(anonymous, authorized)

    response = anonymous.post(
        "/ui/api/pots/use", json={"ref": "default"}, headers={"Origin": origin}
    )

    assert response.status_code == 403, response.text
    assert pots.used == []


# --- DNS rebinding: Host allowlist -----------------------------------------


@pytest.mark.parametrize(
    "host", ["rebound.example:8765", "rebound.example", "127.0.0.1.evil.example"]
)
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/ui"),
        ("GET", "/ui/"),
        ("GET", "/ui/api/pots"),
        ("POST", "/ui/api/handoff"),
    ],
)
def test_a_request_under_a_non_loopback_host_name_is_refused(
    authorized: TestClient, host: str, method: str, path: str
) -> None:
    """A rebound page sends its own name as ``Host``. Refusing it covers the
    shell and assets too, so the explorer never loads under that origin --
    even for a caller holding the token."""
    response = authorized.request(method, path, headers={"Host": host})

    assert response.status_code == 403
    assert "loopback" in response.json()["detail"]


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"])
def test_loopback_host_names_are_accepted(authorized: TestClient, host: str) -> None:
    response = authorized.get("/ui/api/pots", headers={"Host": host})

    assert response.status_code == 200


def test_a_rebound_page_cannot_spend_a_handoff_code(
    app: FastAPI, authorized: TestClient
) -> None:
    """Refused before the shell handler runs: the code is neither redeemed
    under the attacker's name nor burned for the user's own browser."""
    code = _mint(authorized)
    rebound = TestClient(app, base_url=ORIGIN, follow_redirects=False)

    refused = rebound.get(f"/ui?k={code}", headers={"Host": "rebound.example:8765"})
    assert refused.status_code == 403
    assert "set-cookie" not in refused.headers

    browser = TestClient(app, base_url=ORIGIN, follow_redirects=False)
    assert "set-cookie" in browser.get(f"/ui?k={code}").headers


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.1:8765", True),
        ("LOCALHOST:8765", True),
        ("[::1]:8765", True),
        ("", False),
        (None, False),
        ("localhost.:8765", False),
        ("evil.example:8765", False),
        ("127.0.0.1:port", False),
    ],
)
def test_is_loopback_host(value: str | None, expected: bool) -> None:
    assert ui_auth.is_loopback_host(value) is expected


# --- the shell itself ------------------------------------------------------


@pytest.mark.parametrize("path", ["/ui", "/ui/"])
def test_the_spa_shell_loads_without_a_credential(
    anonymous: TestClient, path: str
) -> None:
    """It has to: the page is what runs the handoff. It ships no graph data --
    all of that comes from the authenticated API."""
    response = anonymous.get(path)

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-cache"


@pytest.fixture
def bundled_app(tmp_path, monkeypatch, pots: _Pots) -> FastAPI:
    """The app over a built bundle, so the static mount is exercised too."""
    from potpie.daemon.http.ui import static

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>")
    (tmp_path / "assets" / "app.js").write_text("console.log('explorer')")
    monkeypatch.setattr(static, "frontend_dist_dir", lambda: tmp_path)
    return _app(pots)


def test_the_built_shell_and_its_assets_load_without_a_credential(
    bundled_app: FastAPI,
) -> None:
    browser = TestClient(bundled_app, base_url=ORIGIN, follow_redirects=False)

    shell = browser.get("/ui")
    asset = browser.get("/ui/assets/app.js")

    assert shell.status_code == 200
    assert "id=root" in shell.text
    assert shell.headers["cache-control"] == "no-cache"
    assert asset.status_code == 200
    assert browser.get("/ui/api/pots").status_code == 401


def test_the_built_shell_still_redeems_a_handoff(bundled_app: FastAPI) -> None:
    authorized = TestClient(bundled_app, base_url=ORIGIN, headers=BEARER)
    browser = TestClient(bundled_app, base_url=ORIGIN, follow_redirects=False)

    response = browser.get("/ui", params={"k": _mint(authorized)})

    assert response.status_code == 303
    assert browser.get("/ui/api/pots").status_code == 200


def test_built_assets_are_refused_under_a_rebound_host_name(
    bundled_app: FastAPI,
) -> None:
    browser = TestClient(bundled_app, base_url=ORIGIN)

    response = browser.get(
        "/ui/assets/app.js", headers={"Host": "rebound.example:8765"}
    )

    assert response.status_code == 403


# --- commit history --------------------------------------------------------
#
# A browser session reads history and asks for previews; applying one changes
# the graph and stays with the daemon credential on the CLI.


def test_a_browser_session_reads_commits_and_previews_a_rollback(
    anonymous: TestClient, authorized: TestClient
) -> None:
    _sign_in(anonymous, authorized)

    assert anonymous.get("/ui/api/commits").status_code == 200
    preview = anonymous.post(
        "/ui/api/rollback/preview",
        json=API_CALLS[("POST", "/ui/api/rollback/preview")]["json"],
        headers={"Origin": ORIGIN},
    )

    assert preview.status_code == 200, preview.text
    assert preview.json()["preview"]["preview_id"] == "p1"


def test_the_explorer_has_no_route_that_applies_a_preview(
    app: FastAPI, authorized: TestClient
) -> None:
    assert not any("apply" in path for _method, path in _api_routes(app))
    response = authorized.post(
        "/ui/api/rollback/apply", json={"pot": "pot_1", "preview_id": "p1"}
    )

    assert response.status_code in {404, 405}


def test_a_cross_origin_preview_is_refused_even_with_a_session(
    anonymous: TestClient, authorized: TestClient
) -> None:
    _sign_in(anonymous, authorized)

    response = anonymous.post(
        "/ui/api/rollback/preview",
        json=API_CALLS[("POST", "/ui/api/rollback/preview")]["json"],
        headers={"Origin": "http://evil.example"},
    )

    assert response.status_code == 403


def test_a_preview_body_cannot_name_its_own_actor(authorized: TestClient) -> None:
    body = {**API_CALLS[("POST", "/ui/api/rollback/preview")]["json"]}

    response = authorized.post(
        "/ui/api/rollback/preview", json={**body, "actor": "forged"}
    )

    assert response.status_code == 422


def test_commit_routes_without_a_typed_client_are_unavailable(pots: _Pots) -> None:
    app = build_ui_app(
        pots=pots, graph=_Graph(), backend=_Backend(), bearer_token=TOKEN
    )
    client = TestClient(app, base_url=ORIGIN, headers=BEARER)

    response = client.get("/ui/api/commits")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unavailable"
