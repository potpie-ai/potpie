from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest
from potpie.cli.telemetry import sentry_runtime
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.sentry_runtime import (
    capture_unexpected_cli_error,
    configure_cli_sentry,
)
from potpie.runtime.telemetry import sentry_metrics as sentry_metrics_runtime
from potpie.runtime.telemetry.sentry_settings import SentrySettings


@dataclass
class _Scope:
    tags: dict[str, str] = field(default_factory=dict)
    contexts: dict[str, dict[str, str]] = field(default_factory=dict)

    def __enter__(self) -> "_Scope":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def set_context(self, key: str, value: dict[str, str]) -> None:
        self.contexts[key] = value


class _FakeSentry(ModuleType):
    def __init__(self) -> None:
        super().__init__("sentry_sdk")
        self.init_calls: list[dict[str, object]] = []
        self.captured: list[BaseException] = []
        self.scope = _Scope()

    def init(self, **kwargs: Any) -> None:
        self.init_calls.append(dict(kwargs))

    def new_scope(self) -> _Scope:
        self.scope = _Scope()
        return self.scope

    def capture_exception(self, exc: BaseException) -> None:
        self.captured.append(exc)


@pytest.fixture(autouse=True)
def reset_sentry_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentry_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_enabled", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_sentry_sdk", None)


def test_configure_cli_sentry_uses_privacy_hooks(monkeypatch) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    settings = SentrySettings(
        enabled=True,
        dsn="https://public@example.invalid/1",
        environment="staging",
        release="potpie-cli@test",
        dist="cli-dist",
    )

    configure_cli_sentry(settings)

    assert len(fake.init_calls) == 1
    call = fake.init_calls[0]
    assert call["dsn"] == "https://public@example.invalid/1"
    assert call["send_default_pii"] is False
    assert call["include_local_variables"] is False
    assert call["max_request_body_size"] == "never"
    assert callable(call["before_send"])
    assert callable(call["before_breadcrumb"])


def test_configure_cli_sentry_delegates_to_neutral_runtime(monkeypatch) -> None:
    settings = SentrySettings(
        enabled=True,
        dsn="https://public@example.invalid/1",
        environment="staging",
        release="potpie-cli@test",
        dist="cli-dist",
    )
    configured: list[SentrySettings] = []

    def configure(settings_arg: SentrySettings) -> None:
        configured.append(settings_arg)

    monkeypatch.setattr(sentry_runtime, "configure_metrics", configure)
    monkeypatch.setattr(sentry_runtime, "metrics_configured", lambda: True)

    configure_cli_sentry(settings)

    assert configured == [settings]
    assert sentry_runtime._configured is True


def test_configure_cli_sentry_disabled_does_not_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sentry_sdk", raising=False)
    settings = SentrySettings(
        enabled=False,
        dsn="https://public@example.invalid/1",
        environment="dev",
        release="potpie-cli@test",
        dist=None,
    )

    configure_cli_sentry(settings)

    assert "sentry_sdk" not in sys.modules


def test_capture_unexpected_cli_error_sets_allowlisted_scope(monkeypatch) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    monkeypatch.setattr(sentry_runtime, "_configured", True)
    telemetry = TelemetryContext(
        anonymous_install_id="install_123",
        invocation_id="invoke_456",
        daemon_session_id="daemon_789",
        environment="staging",
        command="daemon",
        subcommand="status",
        output_mode="json",
        cli_version="0.1.0",
        python_version="3.13.0",
        os="darwin",
        arch="arm64",
    )
    monkeypatch.setattr(
        "potpie.cli.telemetry.sentry_runtime.current_telemetry_context",
        lambda: telemetry,
    )

    exc = RuntimeError("boom")
    capture_unexpected_cli_error(
        exc,
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )

    assert fake.captured == [exc]
    assert fake.scope.tags == {
        "arch": "arm64",
        "cli_version": "0.1.0",
        "command": "daemon",
        "error.code": "unexpected_cli_error",
        "error.kind": "unexpected",
        "is_expected": "false",
        "os": "darwin",
        "output_mode": "json",
        "python_version": "3.13.0",
        "service": "potpie-cli",
        "subcommand": "status",
    }
    assert fake.scope.contexts["telemetry"] == {
        "anonymous_install_id": "install_123",
        "daemon_session_id": "daemon_789",
        "invocation_id": "invoke_456",
    }


def test_capture_unexpected_cli_error_skips_when_not_configured(monkeypatch) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    capture_unexpected_cli_error(
        RuntimeError("boom"),
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )

    assert fake.captured == []


def test_capture_unexpected_cli_error_is_nonfatal_without_sentry(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sentry_sdk", raising=False)
    monkeypatch.setattr(sentry_runtime, "_configured", True)

    capture_unexpected_cli_error(
        RuntimeError("boom"),
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )
