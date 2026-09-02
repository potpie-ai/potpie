from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol

from potpie_context_engine.bootstrap.sentry_metrics_runtime import (
    configure_metrics,
    flush,
    metrics_configured,
    set_metric_recorder,
)

from .context import (
    TelemetryContext,
    current_telemetry_context,
)
from .settings import SentrySettings

_configured = False
_settings: SentrySettings | None = None


def spool_metric(
    kind: str,
    name: str,
    value: int | float,
    unit: str | None,
    attributes: dict[str, str | int | float | bool] | None,
) -> None:
    """The CLI's metric recorder: to the spool, shipped later by the flusher.

    Installing this is what keeps ``sentry_sdk`` out of the command's own
    process on the happy path: no import, no ``init``, no envelope at exit.
    """
    from . import spool

    spool.append(
        {
            "kind": "metric",
            "type": kind,
            "name": name,
            "value": value,
            "unit": unit,
            "attributes": dict(attributes or {}),
        }
    )


class _SentryScope(Protocol):
    def set_tag(self, key: str, value: str) -> None: ...

    def set_context(self, key: str, value: dict[str, str]) -> None: ...


def configure_cli_sentry(settings: SentrySettings) -> None:
    """Arm crash reports and route metrics to the spool. Initialises nothing.

    The SDK used to be initialised here on every command, which is where the
    integration probes, the ``init`` work and the envelope at exit came from.
    Now the command's process never touches ``sentry_sdk`` unless it has an
    unexpected error to report: metrics go to the spool through
    :func:`spool_metric`, and :func:`capture_unexpected_cli_error`
    initialises the SDK on demand with the short-lived profile.
    """
    global _configured, _settings
    if not settings.enabled:
        disable_cli_sentry()
        return
    _settings = settings
    set_metric_recorder(spool_metric)
    _configured = True


def disable_cli_sentry() -> None:
    global _configured, _settings
    _configured = False
    _settings = None
    set_metric_recorder(None)


def capture_unexpected_cli_error(
    exc: BaseException,
    *,
    error_code: str,
    error_kind: str,
) -> None:
    if not _configured:
        return
    try:
        if not metrics_configured():
            if _settings is None:
                return
            configure_metrics(_settings, short_lived_process=True)
            if not metrics_configured():
                return
        sentry_sdk = _load_sentry_sdk()
        telemetry = current_telemetry_context()
        with sentry_sdk.new_scope() as scope:
            _bind_base_tags(scope, error_code=error_code, error_kind=error_kind)
            if telemetry is not None:
                _bind_telemetry(scope, telemetry)
            sentry_sdk.capture_exception(exc)
        # The one path where the process waits on the network: an unexpected
        # error is rare and worth a bounded second to report.
        flush(timeout=1.0)
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
