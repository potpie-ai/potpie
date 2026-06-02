"""Tests for Linear OAuth flow in auth_commands."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import typer

from adapters.inbound.cli import auth_commands
from adapters.inbound.cli.callback_server import OAuthCallbackResult


class _ImmediateThread:
    def __init__(self, target=None, daemon=None) -> None:
        self._target = target

    def start(self) -> None:
        if self._target:
            self._target()

    def join(self, timeout=None) -> None:
        return None


def test_run_linear_oauth_flow_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
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
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("verifier", "challenge"))
    monkeypatch.setattr(auth_commands, "webbrowser", MagicMock(open=MagicMock(return_value=True)))
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
    saved: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "save_integration_tokens",
        lambda _provider, tokens: saved.append(tokens),
    )
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_json_blob",
        lambda payload, **kwargs: printed.append(payload),
    )
    auth_commands._run_linear_oauth_flow(force=True)

    assert saved and saved[0]["access_token"] == "access"
    assert printed and printed[-1].get("ok") is True


def test_run_linear_oauth_flow_already_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True, "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _x: False)
    handled: list[str] = []
    monkeypatch.setattr(
        auth_commands,
        "_handle_already_connected",
        lambda provider, status: handled.append(provider),
    )

    auth_commands._run_linear_oauth_flow()

    assert handled == ["linear"]


def test_run_linear_oauth_flow_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
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
    assert "not configured" in captured[0][0].lower()


def test_run_linear_oauth_flow_expired_reauth_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True, "expires_at": 0.0},
    )
    monkeypatch.setattr(auth_commands, "token_needs_refresh", lambda _x: True)
    monkeypatch.setattr(auth_commands, "_try_refresh_linear_session", lambda: False)
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
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
    monkeypatch.setattr(auth_commands, "save_integration_tokens", lambda *_a, **_k: None)
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
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "expected-state")
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
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
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
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
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


def test_run_linear_oauth_flow_exchange_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
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
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_redirect_uri", lambda: "http://localhost:8080/callback")
    monkeypatch.setattr(auth_commands, "get_callback_port", lambda: 8080)
    monkeypatch.setattr(auth_commands, "get_callback_host", lambda: "localhost")
    monkeypatch.setattr(auth_commands, "get_callback_path", lambda: "/callback")
    monkeypatch.setattr(auth_commands.secrets, "token_urlsafe", lambda _n: "state-xyz")
    monkeypatch.setattr(auth_commands, "generate_pkce_pair", lambda: ("v", "c"))
    monkeypatch.setattr(auth_commands.time, "sleep", lambda _s: None)


def test_run_linear_oauth_flow_callback_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_run_linear_oauth_flow_callback_timeout_after_join(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_run_linear_oauth_flow_callback_none_result(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_run_linear_oauth_flow_webbrowser_not_opened(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(auth_commands, "save_integration_tokens", lambda *_a, **_k: None)
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


def test_run_linear_oauth_flow_invalid_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
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
