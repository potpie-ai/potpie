from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli import main as host_cli
from potpie.cli.commands import telemetry as telemetry_cmd
from potpie.cli.telemetry import sentry_runtime
from potpie.cli.telemetry.preferences import (
    load_preferences,
    preferences_path,
)
from potpie.cli.telemetry.product_analytics import ProductAnalyticsSettings
from potpie.cli.telemetry.settings import SentrySettings
from bootstrap import runtime_settings

_TELEMETRY_ENV_NAMES = (
    "POTPIE_ENVIRONMENT",
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
)


@pytest.fixture(autouse=True)
def _clear_telemetry_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in _TELEMETRY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.setattr(runtime_settings, "load_distribution_defaults", lambda: {})
    monkeypatch.setattr(sentry_runtime, "configure_cli_sentry", lambda settings: None)


def _enable_all_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")


def test_telemetry_status_defaults_to_enabled_without_configured_sinks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    result = CliRunner().invoke(host_cli.app, ["telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    assert "Potpie CLI telemetry: enabled" in result.stdout
    assert "Crash reports: disabled" in result.stdout
    assert "Analytics: disabled" in result.stdout
    assert "Install ID:" not in result.stdout
    assert "Identity path:" not in result.stdout
    assert "install_" not in result.stdout
    assert not preferences_path().exists()


def test_telemetry_status_reports_enabled_details_for_active_sinks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)

    result = CliRunner().invoke(host_cli.app, ["telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    assert "Potpie CLI telemetry: enabled" in result.stdout
    assert "Crash reports: anonymous" in result.stdout
    assert "Analytics: anonymous" in result.stdout
    assert "Install ID:" not in result.stdout
    assert "Identity path:" not in result.stdout
    assert "install_" not in result.stdout
    assert not preferences_path().exists()


def test_telemetry_disable_and_enable_persist_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)
    runner = CliRunner()

    disabled = runner.invoke(host_cli.app, ["telemetry", "disable"])
    status_after_disable = runner.invoke(host_cli.app, ["telemetry", "status"])
    enabled = runner.invoke(host_cli.app, ["telemetry", "enable"])
    status_after_enable = runner.invoke(host_cli.app, ["telemetry", "status"])

    assert disabled.exit_code == 0, disabled.stdout
    assert disabled.stdout.strip() == "Potpie CLI telemetry: disabled"
    assert status_after_disable.exit_code == 0, status_after_disable.stdout
    assert status_after_disable.stdout.strip() == "Potpie CLI telemetry: disabled"
    assert enabled.exit_code == 0, enabled.stdout
    assert "Potpie CLI telemetry: enabled" in enabled.stdout
    assert "Crash reports: anonymous" in enabled.stdout
    assert "Analytics: anonymous" in enabled.stdout
    assert status_after_enable.exit_code == 0, status_after_enable.stdout
    assert "Potpie CLI telemetry: enabled" in status_after_enable.stdout
    assert "Crash reports: anonymous" in status_after_enable.stdout
    assert "Analytics: anonymous" in status_after_enable.stdout

    payload = json.loads(preferences_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["enabled"] is True


@pytest.mark.parametrize("contents", ["not json", "[]", "{}"])
def test_telemetry_preferences_fail_closed_for_invalid_existing_file(
    monkeypatch,
    tmp_path,
    contents,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = preferences_path()
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")

    preferences = load_preferences()

    assert preferences.enabled is False


def test_telemetry_preferences_fail_closed_for_unreadable_existing_file(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = preferences_path()
    path.mkdir(parents=True)

    preferences = load_preferences()

    assert preferences.enabled is False


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


def test_telemetry_disable_refreshes_runtime_sinks_with_disabled_settings(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)
    sentry_calls: list[SentrySettings] = []
    analytics_calls: list[ProductAnalyticsSettings] = []
    monkeypatch.setattr(
        telemetry_cmd.sentry_runtime,
        "configure_cli_sentry",
        lambda settings: sentry_calls.append(settings),
    )
    monkeypatch.setattr(
        telemetry_cmd,
        "configure_product_analytics",
        lambda settings: analytics_calls.append(settings),
    )

    result = CliRunner().invoke(host_cli.app, ["telemetry", "disable"])

    assert result.exit_code == 0, result.stdout
    assert sentry_calls[-1].enabled is False
    assert sentry_calls[-1].dsn == "https://public@example.invalid/1"
    assert analytics_calls[-1].enabled is False
    assert analytics_calls[-1].api_key == "phc_test"


def test_telemetry_status_reports_blocked_for_global_env_override(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    result = CliRunner().invoke(host_cli.app, ["telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "Potpie CLI telemetry: blocked"
    assert "blocked by" not in result.stdout.lower()
    assert "Crash reports:" not in result.stdout


def test_telemetry_status_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)

    result = CliRunner().invoke(host_cli.app, ["--json", "telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["telemetry"] == "enabled"
    assert payload["crash_reports"] == "anonymous"
    assert payload["analytics"] == "anonymous"
    assert set(payload) == {"telemetry", "crash_reports", "analytics"}


def test_telemetry_status_json_reports_persisted_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _enable_all_sinks(monkeypatch)
    runner = CliRunner()

    disabled = runner.invoke(host_cli.app, ["telemetry", "disable"])
    result = runner.invoke(host_cli.app, ["--json", "telemetry", "status"])

    assert disabled.exit_code == 0, disabled.stdout
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "telemetry": "disabled",
        "crash_reports": "disabled",
        "analytics": "disabled",
    }


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {},
            {
                "telemetry": "enabled",
                "crash_reports": "disabled",
                "analytics": "disabled",
            },
        ),
        (
            {"POTPIE_SENTRY_DSN": "https://public@example.invalid/1"},
            {
                "telemetry": "enabled",
                "crash_reports": "anonymous",
                "analytics": "disabled",
            },
        ),
        (
            {"POTPIE_POSTHOG_API_KEY": "phc_test"},
            {
                "telemetry": "enabled",
                "crash_reports": "disabled",
                "analytics": "anonymous",
            },
        ),
        (
            {
                "POTPIE_SENTRY_DSN": "https://public@example.invalid/1",
                "POTPIE_POSTHOG_API_KEY": "phc_test",
                "POTPIE_SENTRY_ENABLED": "0",
            },
            {
                "telemetry": "enabled",
                "crash_reports": "disabled",
                "analytics": "anonymous",
            },
        ),
        (
            {
                "POTPIE_SENTRY_DSN": "https://public@example.invalid/1",
                "POTPIE_POSTHOG_API_KEY": "phc_test",
                "POTPIE_PRODUCT_ANALYTICS_ENABLED": "0",
            },
            {
                "telemetry": "enabled",
                "crash_reports": "anonymous",
                "analytics": "disabled",
            },
        ),
        (
            {
                "POTPIE_SENTRY_DSN": "https://public@example.invalid/1",
                "POTPIE_POSTHOG_API_KEY": "phc_test",
                "POTPIE_TELEMETRY_DISABLED": "1",
            },
            {
                "telemetry": "blocked",
                "crash_reports": "blocked",
                "analytics": "blocked",
            },
        ),
    ],
)
def test_telemetry_status_json_reflects_effective_sink_gates(
    monkeypatch,
    tmp_path,
    env,
    expected,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = CliRunner().invoke(host_cli.app, ["--json", "telemetry", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert {
        "telemetry": payload["telemetry"],
        "crash_reports": payload["crash_reports"],
        "analytics": payload["analytics"],
    } == expected
    assert set(payload) == {"telemetry", "crash_reports", "analytics"}


def test_telemetry_preference_write_failure_uses_error_contract_without_refresh(
    monkeypatch,
    tmp_path,
) -> None:
    xdg_file = tmp_path / "not-a-directory"
    xdg_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_file))
    calls: list[str] = []
    monkeypatch.setattr(
        telemetry_cmd,
        "_refresh_runtime_sinks",
        lambda: calls.append("refresh"),
    )

    result = CliRunner().invoke(host_cli.app, ["--json", "telemetry", "disable"])

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "telemetry_preference_write_failed"
    assert payload["message"] == "Could not update telemetry preference."
    assert "settings.json" in payload["detail"]
    assert calls == []


def test_telemetry_command_appears_in_top_level_help() -> None:
    result = CliRunner().invoke(host_cli.app, ["--help"])

    assert result.exit_code == 0, result.stdout
    assert "telemetry" in result.stdout
