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
