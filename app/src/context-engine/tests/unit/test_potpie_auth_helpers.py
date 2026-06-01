"""Unit tests for Potpie CLI auth helpers."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import pytest

from adapters.inbound.cli.auth import potpie as potpie_auth


def test_resolve_potpie_ui_url_uses_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_CLI_UI_BASE_URL", "https://stage.potpie.ai/")
    monkeypatch.delenv("POTPIE_UI_URL", raising=False)

    assert potpie_auth.resolve_potpie_ui_url() == "https://stage.potpie.ai"


def test_build_sign_in_url_encodes_callback_and_state() -> None:
    url = potpie_auth.build_sign_in_url(
        ui_base_url="https://app.potpie.ai/",
        callback_url="http://localhost:9000/callback path",
        state="state value",
    )

    assert url.startswith("https://app.potpie.ai/sign-in?")
    assert "cli_callback=http%3A%2F%2Flocalhost%3A9000%2Fcallback+path" in url
    assert "state=state+value" in url


def test_parse_custom_token_body_accepts_jwt_like_token() -> None:
    result = potpie_auth._parse_custom_token_body(
        json.dumps(
            {
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "expected-state",
            }
        ).encode(),
        expected_state="expected-state",
    )

    assert result.custom_token == "header.payload.signature"
    assert result.firebase_api_key == "firebase-key"


def test_parse_custom_token_body_uses_header_state() -> None:
    result = potpie_auth._parse_custom_token_body(
        json.dumps(
            {
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "wrong-state",
            }
        ).encode(),
        expected_state="expected-state",
        header_state="expected-state",
    )

    assert result.custom_token == "header.payload.signature"


def test_parse_custom_token_body_rejects_wrong_state() -> None:
    with pytest.raises(potpie_auth.PotpieCliAuthError, match="state mismatch"):
        potpie_auth._parse_custom_token_body(
            b'{"custom_token":"header.payload.signature","state":"wrong"}',
            expected_state="expected-state",
        )


@pytest.mark.parametrize(
    "body,match",
    [
        (b"", "empty body"),
        (b"not-json", "not valid JSON"),
        (b"[]", "JSON object"),
        (b'{"custom_token":"header.payload.signature"}', "did not include state"),
        (
            b'{"custom_token":"not-a-jwt","state":"expected-state"}',
            "valid Firebase custom token",
        ),
        (
            b'{"custom_token":"header.payload.signature","state":"expected-state"}',
            "Firebase API config",
        ),
    ],
)
def test_parse_custom_token_body_rejects_invalid_payloads(
    body: bytes,
    match: str,
) -> None:
    with pytest.raises(potpie_auth.PotpieCliAuthError, match=match):
        potpie_auth._parse_custom_token_body(body, expected_state="expected-state")


def _start_callback_server(
    host: str,
    port: int,
    *,
    path: str = "/callback-random",
    state: str = "expected-state",
    allowed_origin: str = "https://potpie.example.com",
) -> tuple[potpie_auth._OneShotCallbackServer, threading.Thread]:
    server = potpie_auth._OneShotCallbackServer(
        (host, port),
        expected_state=state,
        expected_path=path,
        allowed_origin=allowed_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_callback_server(
    server: potpie_auth._OneShotCallbackServer,
    thread: threading.Thread,
) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_callback_server_allows_cors_preflight() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        response = httpx.options(
            f"http://{host}:{port}/callback-random",
            headers={
                "Origin": "https://potpie.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, authorization",
            },
            timeout=5.0,
        )
        assert response.status_code == 204
        assert (
            response.headers.get("access-control-allow-origin")
            == "https://potpie.example.com"
        )
        assert response.headers.get("access-control-allow-private-network") == "true"
    finally:
        _stop_callback_server(server, thread)


def test_callback_server_rejects_wrong_path_and_get() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    server, thread = _start_callback_server(host, port)
    try:
        wrong = httpx.post(
            f"http://{host}:{port}/wrong",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "expected-state",
            },
            timeout=5.0,
        )
        assert wrong.status_code == 404
        get_response = httpx.get(f"http://{host}:{port}/callback-random", timeout=5.0)
        assert get_response.status_code == 405
        assert not server.received.is_set()
    finally:
        _stop_callback_server(server, thread)


def test_wait_for_cli_callback_accepts_post() -> None:
    host = "127.0.0.1"
    port = potpie_auth.pick_callback_port()
    path = "/callback-random"

    def _post_later() -> None:
        time.sleep(0.05)
        httpx.post(
            f"http://{host}:{port}{path}",
            json={
                "custom_token": "header.payload.signature",
                "firebase_api_key": "firebase-key",
                "state": "expected-state",
            },
            timeout=5.0,
        )

    threading.Thread(target=_post_later, daemon=True).start()
    result = potpie_auth.wait_for_cli_callback(
        host=host,
        port=port,
        path=path,
        state="expected-state",
        allowed_origin="https://potpie.example.com",
        timeout_seconds=5.0,
    )
    assert result.custom_token == "header.payload.signature"


def test_wait_for_cli_callback_returns_server_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeServer:
        def __init__(
            self,
            server_address: tuple[str, int],
            *,
            expected_state: str,
            expected_path: str,
            allowed_origin: str,
        ) -> None:
            self.received = threading.Event()
            self.received.set()
            self.result = potpie_auth.CliCallbackResult(
                custom_token="header.payload.signature",
                firebase_api_key="firebase-key",
            )
            self.error = None
            self.closed = False

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            self.closed = True

        def server_close(self) -> None:
            self.closed = True

    monkeypatch.setattr(potpie_auth, "_OneShotCallbackServer", FakeServer)

    result = potpie_auth.wait_for_cli_callback(
        host="127.0.0.1",
        port=12345,
        path="/callback-random",
        state="expected-state",
        allowed_origin="https://potpie.example.com",
        timeout_seconds=5.0,
    )

    assert result.custom_token == "header.payload.signature"
    assert result.firebase_api_key == "firebase-key"


def test_wait_for_cli_callback_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeServer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.received = threading.Event()
            self.result = None
            self.error = None

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    monkeypatch.setattr(potpie_auth, "_OneShotCallbackServer", FakeServer)
    with pytest.raises(potpie_auth.PotpieCliAuthError, match="timed out"):
        potpie_auth.wait_for_cli_callback(
            host="127.0.0.1",
            port=12345,
            path="/callback-random",
            state="expected-state",
            allowed_origin="https://potpie.example.com",
            timeout_seconds=0.01,
        )


def test_run_browser_login_flow_builds_sign_in_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(potpie_auth, "pick_callback_port", lambda: 9123)
    monkeypatch.setattr(potpie_auth.secrets, "token_urlsafe", lambda n: f"tok{n}")
    monkeypatch.setattr(potpie_auth.webbrowser, "open", lambda url: opened.append(url) or True)

    def _wait(**kwargs: Any) -> potpie_auth.CliCallbackResult:
        captured.update(kwargs)
        return potpie_auth.CliCallbackResult(
            custom_token="header.payload.signature",
            firebase_api_key="firebase-key",
        )

    monkeypatch.setattr(potpie_auth, "wait_for_cli_callback", _wait)
    monkeypatch.setenv("POTPIE_CLI_UI_BASE_URL", "https://app.potpie.ai")

    result = potpie_auth.run_browser_login_flow()

    assert result.firebase_api_key == "firebase-key"
    assert opened == [
        "https://app.potpie.ai/sign-in?"
        "cli_callback=http%3A%2F%2Flocalhost%3A9123%2Ftok24&state=tok32"
    ]
    assert captured["host"] == "localhost"
    assert captured["port"] == 9123
    assert captured["path"] == "/tok24"


def test_resolve_potpie_api_url_for_auth_defaults_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "POTPIE_API_URL",
        "POTPIE_BASE_URL",
        "POTPIE_CLI_API_BASE_URL",
        "POTPIE_CLI_BASE_URL",
        "POTPIE_PORT",
        "POTPIE_API_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    assert potpie_auth.resolve_potpie_api_url_for_auth() == "http://localhost:8001"

    monkeypatch.setenv("POTPIE_PORT", "8123")
    assert potpie_auth.resolve_potpie_api_url_for_auth() == "http://127.0.0.1:8123"


def test_fetch_account_me_uses_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, "kwargs": kwargs})
            return httpx.Response(200, json={"email": "dev@example.com"})

    monkeypatch.setattr(potpie_auth.httpx, "Client", FakeClient)

    data = potpie_auth.fetch_account_me(
        api_base_url="https://api.potpie.ai/",
        id_token="id-token",
    )

    assert data == {"email": "dev@example.com"}
    assert calls == [
        {
            "url": "https://api.potpie.ai/api/v1/account/me",
            "kwargs": {
                "headers": {
                    "Accept": "application/json",
                    "Authorization": "Bearer id-token",
                }
            },
        }
    ]


def test_fetch_account_me_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(401, json={"detail": "bad token"})

    monkeypatch.setattr(potpie_auth.httpx, "Client", FakeClient)
    with pytest.raises(potpie_auth.PotpieCliAuthError, match="invalid or expired"):
        potpie_auth.fetch_account_me(api_base_url="https://api.potpie.ai", api_key="sk")

    with pytest.raises(potpie_auth.PotpieCliAuthError, match="No Potpie auth token"):
        potpie_auth.fetch_account_me(api_base_url="https://api.potpie.ai")


def test_revoke_api_key_on_server_handles_not_found_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter([404, 500])

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def delete(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(next(statuses), json={"detail": "server error"})

    monkeypatch.setattr(potpie_auth.httpx, "Client", FakeClient)
    potpie_auth.revoke_api_key_on_server(
        api_base_url="https://api.potpie.ai",
        api_key="sk-test",
    )
    with pytest.raises(potpie_auth.PotpieCliAuthError, match="Failed to revoke"):
        potpie_auth.revoke_api_key_on_server(
            api_base_url="https://api.potpie.ai",
            api_key="sk-test",
        )
