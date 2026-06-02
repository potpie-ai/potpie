"""CLI command coverage for Linear auth subcommands (mocked)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import auth_commands

runner = CliRunner()


def _mock_cli(monkeypatch: pytest.MonkeyPatch, *, json_mode: bool = False) -> None:
    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (json_mode, False))


def test_auth_status_verify_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cli(monkeypatch, json_mode=True)
    monkeypatch.setattr(
        auth_commands,
        "get_integration_status",
        lambda provider: {"provider": provider, "authenticated": True, "auth_type": "oauth"},
    )
    monkeypatch.setattr(
        auth_commands,
        "ensure_valid_integration_tokens",
        lambda _p: (_ for _ in ()).throw(RuntimeError("refresh broke")),
    )
    result = runner.invoke(auth_commands.auth_app, ["status", "--verify"])
    assert result.exit_code == 0
    assert '"verified": false' in result.stdout
    assert "refresh broke" in result.stdout


def test_wait_for_callback_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.inbound.cli.callback_server import OAuthCallbackResult

    monkeypatch.setattr(
        auth_commands,
        "wait_for_oauth_callback",
        lambda **kwargs: OAuthCallbackResult(code="c", state="s"),
    )
    result = auth_commands._wait_for_callback(host="localhost", port=8080, path="/callback")
    assert result.code == "c"


def test_linear_login_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        auth_commands,
        "_run_linear_oauth_flow",
        lambda force=False: called.append(force),
    )
    auth_commands.linear_login(force=True)
    assert called == [True]
