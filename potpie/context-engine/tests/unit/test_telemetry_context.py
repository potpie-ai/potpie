from __future__ import annotations

from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common
from adapters.inbound.cli.telemetry.context import current_telemetry_context
from adapters.inbound.cli.telemetry.identity_store import load_or_create_identity


def test_cli_invocations_share_install_and_daemon_session_ids(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("POTPIE_SENTRY_ENVIRONMENT", "staging")
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.delenv("POTPIE_SENTRY_DSN", raising=False)
    runner = CliRunner()

    first = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    first_ctx = current_telemetry_context()
    second = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    second_ctx = current_telemetry_context()

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert first_ctx is not None
    assert second_ctx is not None
    assert first_ctx.anonymous_install_id == second_ctx.anonymous_install_id
    assert first_ctx.invocation_id != second_ctx.invocation_id
    assert first_ctx.daemon_session_id == second_ctx.daemon_session_id
    assert first_ctx.command == "daemon"
    assert first_ctx.subcommand is None
    assert first_ctx.environment == "staging"
    assert first_ctx.analytics_properties()["environment"] == "staging"
    assert first_ctx.output_mode == "json"
    assert (
        load_or_create_identity().anonymous_install_id == first_ctx.anonymous_install_id
    )


def test_cli_context_does_not_include_command_args(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "pot", "use", "private-name"])
    ctx = current_telemetry_context()

    assert result.exit_code in {_common.EXIT_OK, _common.EXIT_VALIDATION}
    assert ctx is not None
    assert ctx.command == "pot"
    assert ctx.subcommand is None
