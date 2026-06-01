"""Focused tests for Potpie CLI login/logout commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli.auth.firebase_session import FirebaseSession

runner = CliRunner()


def test_login_runs_browser_flow_and_stores_refresh_token(
    monkeypatch,
) -> None:
    store_calls: list[tuple[str, str, str]] = []

    class _CallbackResult:
        custom_token = "header.payload.signature"
        firebase_api_key = "firebase-key"

    monkeypatch.setattr(cli_main, "run_browser_login_flow", lambda: _CallbackResult())
    monkeypatch.setattr(
        cli_main,
        "exchange_custom_token",
        lambda token, *, firebase_api_key=None: FirebaseSession(
            id_token="id-token",
            refresh_token="refresh-token",
            expires_at=123.0,
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "store_potpie_firebase_refresh_token",
        lambda token, *, created_at, firebase_api_key=None: store_calls.append(
            (token, created_at, firebase_api_key or "")
        ),
    )

    result = runner.invoke(cli_main.app, ["login"])

    assert result.exit_code == 0, result.stdout
    assert "Logged in to Potpie successfully." in result.stdout
    assert len(store_calls) == 1
    assert store_calls[0][0] == "refresh-token"
    assert store_calls[0][2] == "firebase-key"


def test_login_api_key_command_keeps_legacy_behavior(monkeypatch) -> None:
    captured: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        cli_main,
        "write_credentials",
        lambda *, api_key, api_base_url=None: captured.append((api_key, api_base_url)),
    )
    monkeypatch.setattr(cli_main, "credentials_path", lambda: Path("/tmp/credentials.json"))

    result = runner.invoke(
        cli_main.app,
        ["login-api-key", "sk-legacy", "--url", "https://api.example.com/"],
    )

    assert result.exit_code == 0, result.stdout
    assert captured == [("sk-legacy", "https://api.example.com/")]
    assert "/tmp/credentials.json" in result.stdout


def test_logout_clears_potpie_auth_only(monkeypatch) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(cli_main, "get_potpie_auth_type", lambda: "firebase_session")
    monkeypatch.setattr(
        cli_main,
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
    monkeypatch.setattr(cli_main, "get_potpie_auth_type", lambda: "api_key")
    monkeypatch.setattr(cli_main, "get_stored_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        cli_main,
        "revoke_api_key_on_server",
        lambda *, api_base_url, api_key: revoked.append(api_key),
    )
    monkeypatch.setattr(
        cli_main,
        "clear_potpie_auth",
        lambda *, clear_api_key=False: cleared.append(clear_api_key),
    )

    result = runner.invoke(cli_main.app, ["logout"])

    assert result.exit_code == 0, result.stdout
    assert revoked == ["sk-test"]
    assert cleared == [True]
