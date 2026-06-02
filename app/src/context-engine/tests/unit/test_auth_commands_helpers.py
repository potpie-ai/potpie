"""Unit tests for auth_commands helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import auth_commands

runner = CliRunner()


def test_build_linear_authorization_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "get_client_id", lambda _p: "client-id")
    monkeypatch.setattr(auth_commands, "get_scopes", lambda _p: "read")
    url = auth_commands._build_linear_authorization_url(
        redirect_uri="http://localhost:8080/callback",
        state="state-1",
        code_challenge="challenge",
    )
    assert "client_id=client-id" in url
    assert "code_challenge=challenge" in url
    assert "linear.app/oauth/authorize" in url


def test_token_is_expired_helper() -> None:
    import time as time_module

    assert auth_commands._token_is_expired(time_module.time() + 3600) is False
    assert auth_commands._token_is_expired(time_module.time() - 10) is True
    assert auth_commands._token_is_expired(None) is False


def test_try_refresh_linear_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    assert auth_commands._try_refresh_linear_session() is True


def test_try_refresh_linear_session_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    assert auth_commands._try_refresh_linear_session() is False


def test_handle_already_connected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    printed: list[dict] = []
    monkeypatch.setattr(
        auth_commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload") or {}),
    )
    auth_commands._handle_already_connected(
        "linear",
        {"auth_type": "oauth", "expires_at": 1.0, "cloud_id": "c1"},
    )
    assert printed[-1].get("already_connected") is True


