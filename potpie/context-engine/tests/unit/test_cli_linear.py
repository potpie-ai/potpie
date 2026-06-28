"""Unit tests for Linear OAuth CLI modules."""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request
import pytest
from potpie.cli.commands._common import set_store
from tests._auth_fakes import InMemoryCredentialStore
from adapters.outbound.cli_auth.callback_server import (
    OAuthCallbackResult,
    _first,
    _oauth_callback_failure_html,
    wait_for_oauth_callback,
)
import base64
import hashlib
from adapters.outbound.cli_auth.pkce import generate_pkce_pair
from unittest.mock import MagicMock, patch
from adapters.outbound.cli_auth import token_exchange as tx
from adapters.outbound.cli_auth import integration_session as session
import typer
from potpie.cli.auth import auth_commands
import json
# --- test_callback_server.py ---


def test_oauth_callback_failure_html_escapes_error() -> None:
    body = _oauth_callback_failure_html('<script>alert("xss")</script>')
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "alert(&quot;xss&quot;)" in body


def test_oauth_callback_failure_html_default_message() -> None:
    body = _oauth_callback_failure_html(None)
    assert "No authorization code received" in body


def test_oauth_callback_result_ok() -> None:
    ok = OAuthCallbackResult(code="abc", state="s1")
    assert ok.ok is True
    bad = OAuthCallbackResult(error="access_denied")
    assert bad.ok is False


def test_first_param_helper() -> None:
    assert _first({"code": ["abc"]}, "code") == "abc"
    assert _first({}, "code") is None
    assert _first({"code": [""]}, "code") is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_oauth_callback_test(
    *,
    port: int,
    hit_url: str,
    path: str = "/callback",
    timeout: float = 5.0,
    on_hit: callable | None = None,
) -> OAuthCallbackResult:
    """Start the callback server first, then hit it (avoids CI race)."""
    result_box: dict[str, OAuthCallbackResult] = {}

    def serve() -> None:
        result_box["result"] = wait_for_oauth_callback(
            host="127.0.0.1", port=port, path=path, timeout=timeout
        )

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    time.sleep(0.15)
    if on_hit is not None:
        on_hit(hit_url)
    else:
        with urllib.request.urlopen(hit_url, timeout=timeout) as resp:
            assert resp.status == 200
    server_thread.join(timeout=timeout + 1.0)
    return result_box["result"]


def test_wait_for_oauth_callback_success() -> None:
    port = _free_port()
    hit_url = f"http://127.0.0.1:{port}/callback?code=auth-code&state=xyz"
    result = _run_oauth_callback_test(port=port, hit_url=hit_url)

    assert result.ok is True
    assert result.code == "auth-code"
    assert result.state == "xyz"


def test_wait_for_oauth_callback_error_query() -> None:
    port = _free_port()
    hit_url = f"http://127.0.0.1:{port}/callback?error=access_denied"
    result = _run_oauth_callback_test(port=port, hit_url=hit_url)

    assert result.ok is False
    assert result.error == "access_denied"


def test_wait_for_oauth_callback_wrong_path_returns_404() -> None:
    port = _free_port()
    hit_url = f"http://127.0.0.1:{port}/callback?code=ignored"

    def on_hit(url: str) -> None:
        try:
            urllib.request.urlopen(url, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    result_box: dict[str, OAuthCallbackResult | BaseException] = {}

    def serve() -> None:
        try:
            result_box["result"] = wait_for_oauth_callback(
                host="127.0.0.1",
                port=port,
                path="/other",
                timeout=0.5,
            )
        except BaseException as exc:
            result_box["result"] = exc

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    time.sleep(0.15)
    on_hit(hit_url)
    server_thread.join(timeout=2.0)

    assert isinstance(result_box["result"], TimeoutError)


# --- test_pkce.py ---


def test_generate_pkce_pair_returns_s256_challenge() -> None:
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) == 64
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected


def test_generate_pkce_pair_produces_unique_values() -> None:
    first = generate_pkce_pair()
    second = generate_pkce_pair()
    assert first != second


# --- test_token_exchange.py ---


@pytest.fixture(autouse=True)
def _linear_client_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "test-linear-client-id")


def test_exchange_authorization_code_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "token_type": "Bearer",
        "scope": "read",
        "expires_in": 3600,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    tokens = tx.exchange_authorization_code(
        "linear",
        code="auth-code",
        code_verifier="verifier",
        redirect_uri="http://localhost:8080/callback",
        http=client,
    )

    assert tokens["access_token"] == "access"
    assert tokens["refresh_token"] == "refresh"
    assert tokens["expires_at"] is not None


def test_exchange_authorization_code_rejects_non_linear() -> None:
    with pytest.raises(ValueError, match="only supported for Linear"):
        tx.exchange_authorization_code(
            "github",  # type: ignore[arg-type]
            code="c",
            code_verifier="v",
        )


def test_exchange_authorization_code_requires_verifier() -> None:
    with pytest.raises(ValueError, match="code_verifier is required"):
        tx.exchange_authorization_code("linear", code="c", code_verifier="")


def test_exchange_authorization_code_http_error() -> None:
    response = MagicMock()
    response.status_code = 400
    response.text = "bad request"
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with pytest.raises(RuntimeError, match="Token exchange failed"):
        tx.exchange_authorization_code(
            "linear",
            code="c",
            code_verifier="v",
            http=client,
        )


def test_refresh_access_token_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": "new-access",
        "token_type": "Bearer",
        "expires_in": 1800,
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    tokens = tx.refresh_access_token("linear", refresh_token="rt", http=client)

    assert tokens["access_token"] == "new-access"
    assert tokens["refresh_token"] == "rt"


def test_refresh_access_token_requires_token() -> None:
    with pytest.raises(ValueError, match="refresh_token is required"):
        tx.refresh_access_token("linear", refresh_token="  ")


def test_exchange_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tx, "get_client_id", lambda _p: "")
    with pytest.raises(ValueError, match="LINEAR_CLIENT_ID"):
        tx.exchange_authorization_code("linear", code="c", code_verifier="v")


def test_refresh_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tx, "get_client_id", lambda _p: "")
    with pytest.raises(ValueError, match="LINEAR_CLIENT_ID"):
        tx.refresh_access_token("linear", refresh_token="rt")


def test_refresh_rejects_non_linear() -> None:
    with pytest.raises(ValueError, match="only supported for Linear"):
        tx.refresh_access_token("github", refresh_token="rt")  # type: ignore[arg-type]


def test_refresh_http_error() -> None:
    response = MagicMock()
    response.status_code = 401
    response.text = "unauthorized"
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with pytest.raises(RuntimeError, match="Token refresh failed"):
        tx.refresh_access_token("linear", refresh_token="rt", http=client)


# --- test_integration_session.py ---


def test_ensure_valid_linear_expired_without_refresh_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = {"access_token": "old", "expires_at": 0.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: expired)
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)

    result = session.ensure_valid_integration_tokens("linear")

    assert result == {}


def test_ensure_valid_linear_refreshes_when_refresh_token_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        "access_token": "old",
        "refresh_token": "rt",
        "expires_at": 0.0,
        "scope": "read",
    }
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(
        session,
        "refresh_access_token",
        lambda _provider, refresh_token: {
            "access_token": "new",
            "refresh_token": refresh_token,
            "expires_at": 9999999999.0,
        },
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        session,
        "save_integration_tokens",
        lambda _provider, tokens: saved.append(tokens),
    )

    result = session.ensure_valid_integration_tokens("linear")

    assert result["access_token"] == "new"
    assert saved and saved[0]["access_token"] == "new"


def test_token_needs_refresh_with_buffer() -> None:
    import time as time_module

    future = time_module.time() + 1000
    assert session.token_needs_refresh(future, buffer_seconds=300) is False
    past = time_module.time() - 10
    assert session.token_needs_refresh(past, buffer_seconds=300) is True
    assert session.token_needs_refresh(None) is False
    assert session.token_needs_refresh("not-a-number") is False


def test_ensure_valid_linear_empty_access_token_returns_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"access_token": "  ", "refresh_token": "rt", "expires_at": 0.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)

    result = session.ensure_valid_integration_tokens("linear")

    assert result == stored


def test_ensure_valid_linear_refresh_preserves_cloud_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        "access_token": "old",
        "refresh_token": "rt",
        "expires_at": 0.0,
        "cloud_id": "cid-1",
        "scope": "read",
    }
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(
        session,
        "refresh_access_token",
        lambda _provider, refresh_token: {
            "access_token": "new",
            "refresh_token": refresh_token,
            "expires_at": 9999999999.0,
        },
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        session,
        "save_integration_tokens",
        lambda _provider, tokens: saved.append(tokens),
    )

    result = session.ensure_valid_integration_tokens("linear")

    assert result["cloud_id"] == "cid-1"
    assert saved and saved[0]["cloud_id"] == "cid-1"


def test_ensure_valid_linear_returns_tokens_when_not_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"access_token": "ok", "expires_at": 9999999999.0}
    monkeypatch.setattr(session, "get_integration_tokens", lambda _p: dict(stored))
    monkeypatch.setattr(session, "token_needs_refresh", lambda _expires_at: False)

    with patch.object(session, "refresh_access_token") as refresh:
        result = session.ensure_valid_integration_tokens("linear")

    assert result == stored
    refresh.assert_not_called()


# --- test_auth_commands_linear_oauth.py ---


class _ImmediateThread:
    def __init__(self, target=None, daemon=None) -> None:
        self._target = target

    def start(self) -> None:
        if self._target:
            self._target()

    def join(self, timeout=None) -> None:
        return None


def test_run_linear_oauth_flow_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    status_reads = iter(
        [
            {"authenticated": False},
            {
                "authenticated": True,
                "login": "Ada",
                "email": "a@example.com",
                "auth_type": "oauth",
            },
        ]
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: next(status_reads),
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(
        auth_commands, "generate_pkce_pair", lambda: ("verifier", "challenge")
    )
    monkeypatch.setattr(
        auth_commands, "webbrowser", MagicMock(open=MagicMock(return_value=True))
    )
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(code="auth-code", state="state-xyz"),
    )
    monkeypatch.setattr(
        auth_commands,
        "exchange_authorization_code",
        lambda *_a, **_k: {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 9999999999.0,
            "scope": "read",
        },
    )
    store = InMemoryCredentialStore()
    set_store(store)
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: printed.append(payload),
    )
    auth_commands._run_linear_oauth_flow(force=True)

    assert store.get_integration_tokens("linear").get("access_token") == "access"
    assert printed and printed[-1].get("ok") is True


def test_run_linear_oauth_flow_already_connected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True, "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _x: False)
    monkeypatch.setattr(
        "adapters.outbound.cli_auth.credentials_store.list_linear_organizations",
        lambda: [{"id": "org-1", "name": "Potpie AI CLI", "key": "potpie-ai-cli"}],
    )

    auth_commands._run_linear_oauth_flow()

    out = capsys.readouterr().out
    assert "already connected" in out.lower()
    assert "--add" in out


def test_run_linear_oauth_flow_missing_client_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "")
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert captured
    assert captured[0][0] == "Linear login unavailable"
    assert "LINEAR_CLIENT_ID" in captured[0][1]


def test_run_linear_oauth_flow_expired_reauth_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True, "expires_at": 0.0},
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _x: True)
    monkeypatch.setattr(auth_commands, "_try_refresh_linear_session", lambda: False)
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(code="c", state="state-xyz"),
    )
    monkeypatch.setattr(
        auth_commands,
        "exchange_authorization_code",
        lambda *_a, **_k: {"access_token": "a", "expires_at": 9999999999.0},
    )
    set_store(InMemoryCredentialStore())
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda message, **kwargs: lines.append(message),
    )
    status_queue = [
        {"authenticated": True, "expires_at": 0.0},
        {"authenticated": True, "login": "Ada", "auth_type": "oauth"},
    ]
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: status_queue.pop(0)
        if status_queue
        else {"authenticated": True, "login": "Ada", "auth_type": "oauth"},
    )

    auth_commands._run_linear_oauth_flow()

    assert any("expired" in line.lower() for line in lines)


def test_run_linear_oauth_flow_state_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(
        auth_commands.secrets, "token_urlsafe", lambda _n: "expected-state"
    )
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(code="c", state="wrong-state"),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert any("State mismatch" in msg for _t, msg in captured)


def test_run_linear_oauth_flow_missing_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(state="state-xyz"),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert any("No authorization code" in msg for _t, msg in captured)


def test_run_linear_oauth_flow_oauth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(
            error="access_denied",
            error_description="user denied",
            state="state-xyz",
        ),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert captured
    assert "Linear OAuth failed" in captured[0][0]


def test_run_linear_oauth_flow_exchange_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(code="code", state="state-xyz"),
    )
    monkeypatch.setattr(
        auth_commands,
        "exchange_authorization_code",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("exchange failed")),
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert any("token exchange" in title.lower() for title, _ in captured)


def _oauth_setup(monkeypatch: pytest.MonkeyPatch, *, json_mode: bool = False) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(
        auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback"
    )
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands.time, "sleep", lambda _s: None)


def test_run_linear_oauth_flow_callback_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oauth_setup(monkeypatch)
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)

    def _boom(**_kwargs: object) -> OAuthCallbackResult:
        raise RuntimeError("port in use")

    monkeypatch.setattr(auth_commands, "_wait_for_callback", _boom)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert captured
    assert "callback failed to start" in captured[0][0].lower()


def test_run_linear_oauth_flow_callback_timeout_after_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oauth_setup(monkeypatch)
    browser = MagicMock()
    monkeypatch.setattr(auth_commands, "webbrowser", browser)
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)

    def _timeout(**_kwargs: object) -> OAuthCallbackResult:
        raise TimeoutError("timed out waiting")

    monkeypatch.setattr(auth_commands, "_wait_for_callback", _timeout)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert any("timed out" in msg.lower() for _t, msg in captured)


def test_run_linear_oauth_flow_callback_none_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oauth_setup(monkeypatch)
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock())
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(auth_commands, "_wait_for_callback", lambda **_kwargs: None)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert captured
    assert any("timed out" in title.lower() for title, _msg in captured)


def test_run_linear_oauth_flow_webbrowser_not_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oauth_setup(monkeypatch, json_mode=False)
    browser = MagicMock()
    browser.open.return_value = False
    monkeypatch.setattr(auth_commands, "webbrowser", browser)
    monkeypatch.setattr(auth_commands.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        auth_commands,
        "_wait_for_callback",
        lambda **_kwargs: OAuthCallbackResult(code="c", state="state-xyz"),
    )
    monkeypatch.setattr(
        auth_commands,
        "exchange_authorization_code",
        lambda *_a, **_k: {"access_token": "a", "expires_at": 9999999999.0},
    )
    set_store(InMemoryCredentialStore())
    lines: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda message, **kwargs: lines.append(message),
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True, "login": "Ada", "auth_type": "oauth"},
    )

    auth_commands._run_linear_oauth_flow(force=True)

    assert any("Could not open a browser" in line for line in lines)


def test_run_linear_oauth_flow_invalid_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")

    def _bad_redirect() -> str:
        raise ValueError("bad redirect")

    monkeypatch.setattr(auth_commands, "get_redirect_uri", _bad_redirect)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    with pytest.raises(typer.Exit):
        auth_commands._run_linear_oauth_flow(force=True)

    assert captured
    assert "redirect" in captured[0][0].lower()


# --- test_auth_commands_linear_json.py ---


def test_linear_refresh_emits_single_json_document(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(auth_commands, "ensure_runtime_environment_loaded", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _provider: {
            "authenticated": True,
            "expires_at": 1,
            "login": "nihit",
            "email": "nihit@example.com",
            "site_name": "Acme",
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _expires_at: True)
    monkeypatch.setattr(auth_commands, "_try_refresh_linear_session", lambda: True)

    auth_commands._run_linear_oauth_flow()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is True
    assert payload["provider"] == "linear"
    assert payload.get("refreshed") is True
