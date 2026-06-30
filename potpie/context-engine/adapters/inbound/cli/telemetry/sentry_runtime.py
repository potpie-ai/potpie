from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol

from bootstrap.sentry_metrics_runtime import configure_metrics, metrics_configured

from .context import (
    TelemetryContext,
    current_telemetry_context,
)
from .settings import SentrySettings

_configured = False


class _SentryScope(Protocol):
    def set_tag(self, key: str, value: str) -> None: ...

    def set_context(self, key: str, value: dict[str, str]) -> None: ...


def configure_cli_sentry(settings: SentrySettings) -> None:
    global _configured
    if not settings.enabled:
        disable_cli_sentry()
        return
    if _configured:
        return
    configure_metrics(settings)
    _configured = metrics_configured()


def disable_cli_sentry() -> None:
    global _configured
    _configured = False


def capture_unexpected_cli_error(
    exc: BaseException,
    *,
    error_code: str,
    error_kind: str,
) -> None:
    if not _configured:
        return
    try:
        sentry_sdk = _load_sentry_sdk()
        telemetry = current_telemetry_context()
        with sentry_sdk.new_scope() as scope:
            _bind_base_tags(scope, error_code=error_code, error_kind=error_kind)
            if telemetry is not None:
                _bind_telemetry(scope, telemetry)
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        return


def _load_sentry_sdk() -> ModuleType:
    return importlib.import_module("sentry_sdk")


def _bind_base_tags(
    scope: _SentryScope,
    *,
    error_code: str,
    error_kind: str,
) -> None:
    scope.set_tag("service", "potpie-cli")
    scope.set_tag("error.code", error_code)
    scope.set_tag("error.kind", error_kind)
    scope.set_tag("is_expected", "false")


def _bind_telemetry(scope: _SentryScope, telemetry: TelemetryContext) -> None:
    for key in (
        "cli_version",
        "python_version",
        "os",
        "arch",
        "command",
        "subcommand",
        "output_mode",
    ):
        value = telemetry.fields().get(key)
        if value is not None:
            scope.set_tag(key, value)
    scope.set_context(
        "telemetry",
        {
            "anonymous_install_id": telemetry.anonymous_install_id,
            "invocation_id": telemetry.invocation_id,
            "daemon_session_id": telemetry.daemon_session_id,
        },
    )
