"""Unit tests for Potpie CLI auth helpers."""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest

from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.commands import _login_impl
from adapters.inbound.cli.auth import firebase_session
from adapters.inbound.cli.auth import potpie as potpie_auth
from adapters.inbound.cli.auth.firebase_session import (
    FirebaseSession,
    FirebaseSessionError,
    exchange_custom_token,
    id_token_expires_at,
    refresh_id_token,
)

runner = CliRunner()


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


def test_reserve_callback_socket_returns_listening_socket() -> None:
    listener_socket, port = potpie_auth.reserve_callback_socket()
    try:
        host, bound_port = listener_socket.getsockname()[:2]
        assert bound_port == port
        probe = socket.create_connection((host, port), timeout=1.0)
        probe.close()
    finally:
        listener_socket.close()


def test_wait_for_cli_callback_accepts_post() -> None:
    host = "127.0.0.1"
    listener_socket, port = potpie_auth.reserve_callback_socket()
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
        listener_socket=listener_socket,
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
            listener_socket: socket.socket | None = None,
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
    listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        monkeypatch.setattr(
            potpie_auth,
            "reserve_callback_socket",
            lambda: (listener_socket, 9123),
        )
        monkeypatch.setattr(potpie_auth.secrets, "token_urlsafe", lambda n: f"tok{n}")
        monkeypatch.setattr(
            potpie_auth.webbrowser,
            "open",
            lambda url: opened.append(url) or True,
        )

        def _wait(**kwargs: Any) -> potpie_auth.CliCallbackResult:
            captured.update(kwargs)
            return potpie_auth.CliCallbackResult(
                custom_token="header.payload.signature",
                firebase_api_key="firebase-key",
            )

        monkeypatch.setattr(potpie_auth, "wait_for_cli_callback", _wait)
        monkeypatch.delenv("POTPIE_UI_URL", raising=False)
        monkeypatch.delenv("POTPIE_CLI_APP_BASE_URL", raising=False)
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
        assert captured["listener_socket"] is listener_socket
    finally:
        listener_socket.close()


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


def test_fetch_account_me_wraps_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("GET", url)
            raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setattr(potpie_auth.httpx, "Client", FakeClient)

    with pytest.raises(potpie_auth.PotpieCliAuthError, match="Failed to fetch account: connection failed"):
        potpie_auth.fetch_account_me(api_base_url="https://api.potpie.ai", api_key="sk")


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


def test_revoke_api_key_on_server_wraps_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def delete(self, url: str, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("DELETE", url)
            raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setattr(potpie_auth.httpx, "Client", FakeClient)

    with pytest.raises(
        potpie_auth.PotpieCliAuthError,
        match="Failed to revoke API key on server: connection failed",
    ):
        potpie_auth.revoke_api_key_on_server(
            api_base_url="https://api.potpie.ai",
            api_key="sk-test",
        )

# Firebase session helper coverage

def test_exchange_custom_token_calls_firebase(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, "kwargs": kwargs})
            return httpx.Response(
                200,
                json={
                    "idToken": "id-token",
                    "refreshToken": "refresh-token",
                    "expiresIn": "3600",
                },
            )

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    session = exchange_custom_token("header.payload.signature", firebase_api_key="key")

    assert session.id_token == "id-token"
    assert session.refresh_token == "refresh-token"
    assert calls[0]["url"].endswith("accounts:signInWithCustomToken?key=key")
    assert calls[0]["kwargs"]["json"] == {
        "token": "header.payload.signature",
        "returnSecureToken": True,
    }


def test_resolve_firebase_api_key_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "POTPIE_FIREBASE_API_KEY",
        "GOOGLE_IDENTITY_TOOL_KIT_KEY",
        "FIREBASE_API_KEY",
        "NEXT_PUBLIC_FIREBASE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_FIREBASE_API_KEY", "public-key")

    assert firebase_session.resolve_firebase_api_key() == "public-key"


def test_resolve_firebase_api_key_errors_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "POTPIE_FIREBASE_API_KEY",
        "GOOGLE_IDENTITY_TOOL_KIT_KEY",
        "FIREBASE_API_KEY",
        "NEXT_PUBLIC_FIREBASE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(FirebaseSessionError, match="Firebase API key missing"):
        firebase_session.resolve_firebase_api_key()


def test_exchange_custom_token_rejects_non_jwt_like_token() -> None:
    with pytest.raises(FirebaseSessionError, match="not JWT-like"):
        exchange_custom_token("not-a-jwt", firebase_api_key="key")


def test_exchange_custom_token_errors_on_firebase_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="custom token exchange failed"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_exchange_custom_token_requires_id_and_refresh_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"idToken": "id-token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="missing idToken or refreshToken"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_exchange_custom_token_wraps_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="request failed: connection failed"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_exchange_custom_token_wraps_json_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, text="not json")

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="response parsing failed"):
        exchange_custom_token("header.payload.signature", firebase_api_key="key")


def test_refresh_id_token_form_encodes_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, "kwargs": kwargs})
            return httpx.Response(
                200,
                json={
                    "id_token": "new-id-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": "3600",
                },
            )

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    session = refresh_id_token("refresh+token/with=symbols", firebase_api_key="key")

    assert session.id_token == "new-id-token"
    assert session.refresh_token == "new-refresh-token"
    assert calls[0]["url"].endswith("/token?key=key")
    assert (
        "refresh_token=refresh%2Btoken%2Fwith%3Dsymbols"
        in calls[0]["kwargs"]["content"]
    )


def test_refresh_id_token_errors_on_firebase_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad refresh"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="token refresh failed"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_requires_response_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"id_token": "id-token"})

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="missing id_token or refresh_token"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_wraps_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="request failed: connection failed"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_wraps_json_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, text="not json")

    monkeypatch.setattr(firebase_session.httpx, "Client", FakeClient)

    with pytest.raises(FirebaseSessionError, match="response parsing failed"):
        refresh_id_token("refresh-token", firebase_api_key="key")


def test_refresh_id_token_requires_refresh_token() -> None:
    with pytest.raises(FirebaseSessionError, match="refresh token is missing"):
        refresh_id_token(" ", firebase_api_key="key")


def test_id_token_expires_at_reads_jwt_exp() -> None:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 12345}).encode()).decode()
    token = f"header.{payload.rstrip('=')}.signature"

    assert id_token_expires_at(token) == 12345.0


def test_id_token_expires_at_falls_back_for_invalid_token() -> None:
    before = time.time()
    expires_at = id_token_expires_at("not-a-token")

    assert before + 3590 <= expires_at <= before + 3610


# CLI auth command coverage

def test_login_runs_browser_flow_and_stores_refresh_token(
    monkeypatch,
) -> None:
    store_calls: list[tuple[str, str, str]] = []
    id_token_calls: list[str] = []

    class _CallbackResult:
        custom_token = "header.payload.signature"
        firebase_api_key = "firebase-key"

    monkeypatch.setattr(_login_impl, "run_browser_login_flow", lambda: _CallbackResult())
    monkeypatch.setattr(
        _login_impl,
        "exchange_custom_token",
        lambda token, *, firebase_api_key=None: FirebaseSession(
            id_token="id-token",
            refresh_token="refresh-token",
            expires_at=123.0,
        ),
    )
    monkeypatch.setattr(
        _login_impl,
        "store_potpie_firebase_refresh_token",
        lambda token, *, created_at, firebase_api_key=None: store_calls.append(
            (token, created_at, firebase_api_key or "")
        ),
    )
    monkeypatch.setattr(
        _login_impl,
        "store_potpie_firebase_id_token",
        id_token_calls.append,
    )

    result = runner.invoke(cli_main.app, ["login"])

    assert result.exit_code == 0, result.stdout
    assert "Logged in to Potpie successfully." in result.stdout
    assert len(store_calls) == 1
    assert store_calls[0][0] == "refresh-token"
    assert store_calls[0][2] == "firebase-key"
    assert id_token_calls == ["id-token"]


def test_login_api_key_command_stores_api_key_securely(monkeypatch) -> None:
    stored_keys: list[tuple[str, str]] = []
    stored_urls: list[str | None] = []
    monkeypatch.setattr(
        _login_impl,
        "store_potpie_api_key",
        lambda token, *, created_at: stored_keys.append((token, created_at)),
    )
    monkeypatch.setattr(
        _login_impl,
        "write_api_base_url",
        lambda api_base_url: stored_urls.append(api_base_url),
    )

    result = runner.invoke(
        cli_main.app,
        ["login", "--api-key", "sk-legacy", "--url", "https://api.example.com/"],
    )

    assert result.exit_code == 0, result.stdout
    assert len(stored_keys) == 1
    assert stored_keys[0][0] == "sk-legacy"
    assert stored_urls == ["https://api.example.com/"]
    assert "Saved API key to keyring" in result.stdout


def test_logout_clears_potpie_auth_only(monkeypatch) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(_login_impl, "get_potpie_auth_type", lambda: "potpie")
    monkeypatch.setattr(
        _login_impl,
        "clear_potpie_auth",
        lambda *, clear_api_key=False: cleared.append(clear_api_key),
    )

    result = runner.invoke(cli_main.app, ["logout"])

    assert result.exit_code == 0, result.stdout
    assert cleared == [False]
    assert "Logged out of Potpie." in result.stdout


def test_logout_revokes_api_key_when_auth_type_is_api_key(monkeypatch) -> None:
    revoked: list[str] = []
    cleared: list[bool] = []
    monkeypatch.setattr(_login_impl, "get_potpie_auth_type", lambda: "api_key")
    monkeypatch.setattr(_login_impl, "get_stored_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        _login_impl,
        "revoke_api_key_on_server",
        lambda *, api_base_url, api_key: revoked.append(api_key),
    )
    monkeypatch.setattr(
        _login_impl,
        "clear_potpie_auth",
        lambda *, clear_api_key=False: cleared.append(clear_api_key),
    )

    result = runner.invoke(cli_main.app, ["logout"])

    assert result.exit_code == 0, result.stdout
    assert revoked == ["sk-test"]
    assert cleared == [True]


