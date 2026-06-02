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
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Team",
            "site_url": "https://team.atlassian.net",
            "auth_type": "api_token",
        },
    )
    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0
    assert "jira: authenticated" in result.stdout
    assert "confluence: not authenticated" in result.stdout


def test_jira_login_help() -> None:
    result = runner.invoke(auth_commands.auth_app, ["jira", "login", "--help"])
    assert result.exit_code == 0
    assert "api-token" in result.stdout.lower() or "token" in result.stdout.lower()


def test_auth_status_human_verify_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "site_name": "Team",
            "site_url": "https://team.atlassian.net",
            "expires_at": 12345.0,
            "cloud_id": "cloud-1",
            "token_storage": "keychain",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {"api_token": "tok"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (False, "gateway down"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify failed" in result.stdout
    assert "gateway down" in result.stdout
    assert "expires_at=" in result.stdout
    assert "cloud_id=" in result.stdout


def test_auth_status_human_verify_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "login": "Ada",
            "email": "ada@example.com",
            "auth_type": "api_token",
        },
    )
    monkeypatch.setattr(
        auth_commands,
        "get_integration_tokens",
        lambda _p: {"api_token": "tok", "email": "ada@example.com", "site_url": "https://x.net"},
    )
    monkeypatch.setattr(
        auth_commands,
        "verify_integration_access",
        lambda _p, _c: (True, "ok (Ada)"),
    )

    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])

    assert result.exit_code == 0
    assert "verify=ok (Ada)" in result.stdout
