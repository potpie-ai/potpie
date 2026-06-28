from __future__ import annotations

import json

from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import telemetry as telemetry_cmd
from adapters.inbound.cli.telemetry.identity_store import identity_path
from adapters.inbound.cli.telemetry.preferences import preferences_path


def test_telemetry_status_defaults_to_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    result = CliRunner().invoke(host_cli.app, ["telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    assert "Potpie CLI telemetry: enabled" in result.stdout
    assert "Crash reports: anonymous" in result.stdout
    assert "Analytics: anonymous" in result.stdout
    assert "Install ID: install_" in result.stdout
    assert "Identity path:" in result.stdout
    assert identity_path().name in result.stdout
    assert not preferences_path().exists()


def test_telemetry_disable_and_enable_persist_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    runner = CliRunner()

    disabled = runner.invoke(host_cli.app, ["telemetry", "disable"])
    status_after_disable = runner.invoke(host_cli.app, ["telemetry", "status"])
    enabled = runner.invoke(host_cli.app, ["telemetry", "enable"])
    status_after_enable = runner.invoke(host_cli.app, ["telemetry", "status"])

    assert disabled.exit_code == 0, disabled.stdout
    assert "Potpie CLI telemetry: disabled" in disabled.stdout
    assert status_after_disable.exit_code == 0, status_after_disable.stdout
    assert "Potpie CLI telemetry: disabled" in status_after_disable.stdout
    assert enabled.exit_code == 0, enabled.stdout
    assert "Potpie CLI telemetry: enabled" in enabled.stdout
    assert status_after_enable.exit_code == 0, status_after_enable.stdout
    assert "Potpie CLI telemetry: enabled" in status_after_enable.stdout

    payload = json.loads(preferences_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["enabled"] is True


def test_telemetry_preference_commands_refresh_runtime_sinks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    calls: list[str] = []
    monkeypatch.setattr(
        telemetry_cmd,
        "_refresh_runtime_sinks",
        lambda: calls.append("refresh"),
    )
    runner = CliRunner()

    disabled = runner.invoke(host_cli.app, ["telemetry", "disable"])
    enabled = runner.invoke(host_cli.app, ["telemetry", "enable"])

    assert disabled.exit_code == 0, disabled.stdout
    assert enabled.exit_code == 0, enabled.stdout
    assert calls == ["refresh", "refresh"]


def test_telemetry_status_reports_blocked_for_global_env_override(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    result = CliRunner().invoke(host_cli.app, ["telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    assert "Potpie CLI telemetry: blocked" in result.stdout
    assert "blocked by" not in result.stdout.lower()


def test_telemetry_status_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    result = CliRunner().invoke(host_cli.app, ["--json", "telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["telemetry"] == "enabled"
    assert payload["crash_reports"] == "anonymous"
    assert payload["analytics"] == "anonymous"
    assert payload["install_id"].startswith("install_")
    assert payload["identity_path"] == str(identity_path())


def test_telemetry_command_appears_in_top_level_help() -> None:
    result = CliRunner().invoke(host_cli.app, ["--help"])

    assert result.exit_code == 0, result.stdout
    assert "telemetry" in result.stdout
