"""Auth status and provider subcommand tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import auth_commands

runner = CliRunner()


def test_auth_status_human_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Acme",
            "auth_type": "oauth",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "linear: authenticated" in result.stdout


def test_auth_status_verify_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": True,
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _provider: {"access_token": "tok", "expires_at": 9999999999.0},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _provider, _creds: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0, result.stdout
    assert '"verified": true' in result.stdout


def test_auth_status_human_verify_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Acme",
            "expires_at": 12345.0,
            "token_storage": "keychain",
            "auth_type": "oauth",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "Linear API request failed"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify failed" in result.stdout
    assert "Linear API request failed" in result.stdout.replace("\n", "")
    assert "expires_at=" in result.stdout


def test_auth_status_human_verify_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "login": "Ada",
            "auth_type": "oauth",
            "expires_at": 9999999999.0,
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: {"access_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify=ok (Ada)" in result.stdout
