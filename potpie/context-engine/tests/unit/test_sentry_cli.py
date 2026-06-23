from __future__ import annotations

import json
from typing import Final, NamedTuple

import pytest
import typer
from typer.testing import CliRunner

from context_engine.adapters.inbound.cli import host_cli
from context_engine.adapters.inbound.cli.commands import _common
from context_engine.adapters.inbound.cli.telemetry.settings import SentrySettings
from context_engine.adapters.inbound.cli.telemetry_context import (
    current_telemetry_context,
    load_anonymous_install_id,
)
from context_engine.adapters.inbound.cli.telemetry.identity_store import identity_path
from context_engine.domain.errors import CapabilityNotImplemented, ContextEngineDisabled, PotNotFound

_SAFE_CLI_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "arch",
        "cli_version",
        "command",
        "error_code",
        "os",
        "output_mode",
        "result",
        "subcommand",
    }
)


class _MetricCall(NamedTuple):
    name: str
    value: int | float
    unit: str | None
    attributes: dict[str, str | int | float | bool]


class _FakeMetricsRuntime:
    def __init__(self) -> None:
        self.count_calls: list[_MetricCall] = []
        self.distribution_calls: list[_MetricCall] = []
        self.flush_calls: list[float] = []

    def count(
        self,
        name: str,
        value: int | float = 1,
        *,
        unit: str | None = None,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        self.count_calls.append(_MetricCall(name, value, unit, dict(attributes or {})))

    def distribution(
        self,
        name: str,
        value: int | float,
        *,
        unit: str | None = None,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        self.distribution_calls.append(
            _MetricCall(name, value, unit, dict(attributes or {}))
        )

    def flush(self, timeout: float = 2.0) -> None:
        self.flush_calls.append(timeout)


@pytest.fixture
def fake_metrics(monkeypatch: pytest.MonkeyPatch) -> _FakeMetricsRuntime:
    fake = _FakeMetricsRuntime()
    monkeypatch.setattr("context_engine.bootstrap.sentry_metrics_runtime.count", fake.count)
    monkeypatch.setattr(
        "context_engine.bootstrap.sentry_metrics_runtime.distribution", fake.distribution
    )
    monkeypatch.setattr("context_engine.bootstrap.sentry_metrics_runtime.flush", fake.flush)
    return fake


@pytest.fixture(autouse=True)
def reset_cli_state() -> None:
    _common.set_json(False)
    _common.set_verbose(False)


def test_in_process_cli_invocations_share_install_and_daemon_session_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "context-home"))
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
    in_process_daemon_session_id = first_ctx.daemon_session_id
    assert second_ctx.daemon_session_id == in_process_daemon_session_id
    assert first_ctx.command == "daemon"
    assert load_anonymous_install_id(tmp_path) == first_ctx.anonymous_install_id
    assert identity_path().is_file()
    assert not (tmp_path / "context-home" / "telemetry" / "identity.json").exists()


def test_cli_root_configures_sentry_errors_and_metrics_with_one_settings_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    settings = SentrySettings(
        enabled=True,
        dsn="https://public@example.invalid/1",
        environment="test",
        release="potpie-cli@test",
        dist=None,
    )
    loaded: list[None] = []
    error_settings: list[SentrySettings] = []
    metric_settings: list[SentrySettings] = []
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.settings.load_sentry_settings",
        lambda: loaded.append(None) or settings,
    )
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.sentry_runtime.configure_cli_sentry",
        error_settings.append,
    )
    monkeypatch.setattr(
        "context_engine.bootstrap.sentry_metrics_runtime.configure_metrics", metric_settings.append
    )
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == 0, result.output
    assert loaded == [None]
    assert error_settings == [settings]
    assert metric_settings == [settings]


def test_expected_contract_error_does_not_capture_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise ValueError("bad input")

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert captured == []


def test_contract_records_success_metrics_without_command_metadata(
    monkeypatch: pytest.MonkeyPatch,
    fake_metrics: _FakeMetricsRuntime,
) -> None:
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.context.current_telemetry_context", lambda: None
    )

    with _common.contract():
        pass

    _assert_metric_outcome(fake_metrics, result="ok", error_code="none")
    attrs = fake_metrics.count_calls[0].attributes
    assert "command" not in attrs
    assert "subcommand" not in attrs
    assert attrs == {"error_code": "none", "result": "ok"}


@pytest.mark.parametrize(
    ("raised", "result", "error_code", "exit_code"),
    [
        (ValueError("bad input"), "validation_error", "validation_error", 1),
        (
            CapabilityNotImplemented("graph.inspect", detail="not wired"),
            "not_implemented",
            "not_implemented",
            2,
        ),
        (PotNotFound("missing pot"), "pot_not_found", "pot_not_found", 1),
        (ContextEngineDisabled("disabled"), "unavailable", "unavailable", 2),
        (typer.Exit(code=7), "exit", "exit", 7),
    ],
)
def test_contract_records_expected_error_metrics(
    raised: BaseException,
    result: str,
    error_code: str,
    exit_code: int,
    fake_metrics: _FakeMetricsRuntime,
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise raised

    assert exc_info.value.exit_code == exit_code
    _assert_metric_outcome(fake_metrics, result=result, error_code=error_code)


def test_contract_preserves_expected_typer_exit_from_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_metrics: _FakeMetricsRuntime,
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            _common.fail(
                code="no_active_pot",
                message="No active pot.",
                next_action="run 'potpie setup'",
            )

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert payload["code"] == "no_active_pot"
    assert captured == []
    _assert_metric_outcome(fake_metrics, result="exit", error_code="exit")


def test_unexpected_contract_error_is_captured_and_rendered_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_metrics: _FakeMetricsRuntime,
) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise RuntimeError("boom with /Users/dsantra/private/path")

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert payload["code"] == "unexpected_cli_error"
    assert payload["message"] == "Unexpected internal error."
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]
    _assert_metric_outcome(
        fake_metrics, result="unexpected", error_code="unexpected_cli_error"
    )


def test_cli_telemetry_identity_write_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    xdg_file = tmp_path / "not-a-directory"
    xdg_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_file))
    install_id = load_anonymous_install_id()
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert install_id.startswith("install_")
    assert result.exit_code == 0, result.output
    assert "NotADirectoryError" not in result.output


def _assert_metric_outcome(
    fake_metrics: _FakeMetricsRuntime, *, result: str, error_code: str
) -> None:
    assert len(fake_metrics.count_calls) == 1
    count_call = fake_metrics.count_calls[0]
    assert count_call.name == "ce.cli.invocations_total"
    assert count_call.value == 1
    assert count_call.unit is None
    assert count_call.attributes["error_code"] == error_code
    assert count_call.attributes["result"] == result
    assert set(count_call.attributes).issubset(_SAFE_CLI_ATTRS)
    assert len(fake_metrics.distribution_calls) == 1
    duration = fake_metrics.distribution_calls[0]
    assert duration.name == "ce.cli.duration_ms"
    assert isinstance(duration.value, (int, float))
    assert duration.value >= 0
    assert duration.unit == "millisecond"
    assert duration.attributes == count_call.attributes
    assert fake_metrics.flush_calls == [2.0]
    assert set(duration.attributes).issubset(_SAFE_CLI_ATTRS)
