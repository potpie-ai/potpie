"""Shared auth command registration, output, and logout behavior."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from potpie.auth.adapters.integration_verify import verify_integration_access
from potpie.cli import main as cli_main
from potpie.cli.auth import auth_commands
from potpie.cli.commands._common import set_store
from tests._auth_cli_fixtures import (
    auth_cli as auth_cli,
    auth_json_cli as auth_json_cli,
)
from tests._auth_fakes import InMemoryCredentialStore


runner = CliRunner()


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


def test_auth_logout_unknown_provider(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_flags_delegates_to_common(monkeypatch: pytest.MonkeyPatch) -> None:
    from potpie.cli.commands import _common

    monkeypatch.setattr(_common, "is_json", lambda: True)
    monkeypatch.setattr(_common, "is_verbose", lambda: True)
    assert auth_commands._flags() == (True, True)


def test_auth_help_is_wired_into_main_cli() -> None:
    result = runner.invoke(cli_main.app, ["auth", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Integration auth status" in result.stdout
    assert "Deprecated" in result.stdout


@pytest.mark.parametrize("provider", ["linear", "jira"])
def test_auth_revoke_delegates_to_logout(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )
    set_store(InMemoryCredentialStore())

    result = runner.invoke(auth_commands.auth_app, ["revoke", provider])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout


@pytest.mark.parametrize("provider", ["linear", "jira"])
def test_auth_logout_clear_failure(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from potpie.auth.adapters.credentials_store import (
        ProviderCredentialError,
    )

    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, True))
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda _p: {"authenticated": True},
    )

    store = InMemoryCredentialStore()

    def _fail_clear(_provider: str) -> None:
        raise ProviderCredentialError("keychain broke")

    store.clear_integration_tokens = _fail_clear  # type: ignore[method-assign]
    set_store(store)

    result = runner.invoke(auth_commands.auth_app, ["logout", provider])

    assert result.exit_code == 4


def test_auth_status_json_linear(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "linear",
            "auth_type": "oauth" if provider == "linear" else "api_token",
        },
    )

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert '"integrations"' in result.stdout
    assert '"linear"' in result.stdout


def test_auth_status_json_atlassian(
    auth_json_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {
            "provider": provider,
            "authenticated": provider == "jira",
            "auth_type": "api_token",
        },
    )

    result = runner.invoke(auth_commands.auth_app, ["status"])

    assert result.exit_code == 0, result.stdout
    assert '"integrations"' in result.stdout
    assert '"jira"' in result.stdout


def test_auth_status_human_output_linear(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert "github: not authenticated" in result.stdout


def test_auth_status_human_output_atlassian(
    auth_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_verify_integration_access_unknown_provider() -> None:
    ok, message = verify_integration_access("slack", {})  # type: ignore[arg-type]
    assert ok is False
    assert "unknown provider" in message
