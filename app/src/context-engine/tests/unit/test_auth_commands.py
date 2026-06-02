"""Generic CLI auth command foundation."""

from __future__ import annotations

import typer
import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import auth_commands
from adapters.inbound.cli import main as cli_main

runner = CliRunner()


def test_auth_help_is_wired_into_main_cli() -> None:
    result = runner.invoke(cli_main.app, ["auth", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Authenticate CLI integrations." in result.stdout


def test_register_provider_app_normalizes_name(monkeypatch) -> None:
    calls: list[tuple[typer.Typer, str]] = []

    def add_typer(provider_app: typer.Typer, *, name: str) -> None:
        calls.append((provider_app, name))

    monkeypatch.setattr(auth_commands.auth_app, "add_typer", add_typer)

    provider_app = typer.Typer()
    auth_commands.register_provider_app(" Example ", provider_app)

    assert calls == [(provider_app, "example")]


def test_register_provider_app_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="provider app name must be non-empty"):
        auth_commands.register_provider_app(" ", typer.Typer())


def test_esc_handles_none_and_markup() -> None:
    from rich.markup import escape

    assert auth_commands._esc(None) == ""
    raw = "[bold]x[/bold]"
    assert auth_commands._esc(raw) == escape(raw)


def test_auth_status_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "auth_type": "oauth" if provider == "linear" else "api_token",
        },
    )
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert '"integrations"' in result.stdout
    assert '"linear"' in result.stdout


def test_auth_logout_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    result = runner.invoke(auth_commands.auth_app, ["logout", "unknown"])

    assert result.exit_code == 1
    assert captured
    assert "Unknown provider" in captured[0][0]


def test_flags_delegates_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adapters.inbound.cli.main._flags",
        lambda: (True, True),
    )
    assert auth_commands._flags() == (True, True)


def test_token_is_expired_invalid_expires_at() -> None:
    assert auth_commands._token_is_expired("not-a-number") is False
    assert auth_commands._token_is_expired(0.0) is True


def test_auth_logout_not_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": False},
    )
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        auth_commands,
        "emit_error",
        lambda title, message, **kwargs: captured.append((title, message)),
    )

    result = runner.invoke(auth_commands.auth_app, ["logout", "linear"])

    assert result.exit_code == 1
    assert captured
    assert "not authenticated" in captured[0][0]


def test_auth_revoke_delegates_to_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (True, False))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    monkeypatch.setattr(auth_commands, "clear_integration_tokens", lambda _p: None)

    result = runner.invoke(auth_commands.auth_app, ["revoke", "linear"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout


def test_auth_logout_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.inbound.cli.credentials_store import ProviderCredentialError

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, True))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )

    def _fail_clear(_provider: str) -> None:
        raise ProviderCredentialError("keychain broke")

    monkeypatch.setattr(auth_commands, "clear_integration_tokens", _fail_clear)

    result = runner.invoke(auth_commands.auth_app, ["logout", "linear"])

    assert result.exit_code == 1


