from __future__ import annotations

import json

from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common
from adapters.inbound.cli.telemetry_context import current_telemetry_context
from domain.errors import CapabilityNotImplemented


class _CrashingDaemon:
    def status(self) -> dict[str, str]:
        raise RuntimeError("daemon crashed with /Users/dsantra/private/path")

    def stop(self) -> dict[str, str]:
        raise CapabilityNotImplemented(
            "host.daemon.stop",
            recommended_next_action="host runs in-process; nothing to stop",
        )


class _FakeHost:
    daemon = _CrashingDaemon()


def test_daemon_unexpected_failure_is_captured_with_session_id(
    monkeypatch,
    tmp_path,
) -> None:
    captured: list[tuple[str, str, str | None]] = []
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (error_code, error_kind, current_telemetry_context().daemon_session_id)
        ),
    )
    _common.set_host(_FakeHost())
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    payload = json.loads(result.stdout)
    assert result.exit_code == _common.EXIT_VALIDATION
    assert payload["code"] == "unexpected_cli_error"
    assert captured == [
        (
            "unexpected_cli_error",
            "unexpected",
            current_telemetry_context().daemon_session_id,
        )
    ]


def test_daemon_expected_not_implemented_is_not_captured(
    monkeypatch,
    tmp_path,
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    _common.set_host(_FakeHost())
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "stop"])

    payload = json.loads(result.stdout)
    assert result.exit_code == _common.EXIT_UNAVAILABLE
    assert payload["code"] == "not_implemented"
    assert captured == []
