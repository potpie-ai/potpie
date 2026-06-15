from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol

from .context import (
    TelemetryContext,
    current_telemetry_context,
)
from .sentry_privacy import (
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)
from .settings import SentrySettings

_configured = False


class _SentryScope(Protocol):
    def set_tag(self, key: str, value: str) -> None: ...

    def set_context(self, key: str, value: dict[str, str]) -> None: ...


def configure_cli_sentry(settings: SentrySettings) -> None:
    global _configured
    if not settings.enabled or settings.dsn is None or _configured:
        return
    try:
        sentry_sdk = _load_sentry_sdk()
        sentry_sdk.init(
            dsn=settings.dsn,
            environment=settings.environment,
            release=settings.release,
            dist=settings.dist,
            send_default_pii=False,
            include_local_variables=False,
            max_request_body_size="never",
            before_send=scrub_sentry_event,
            before_breadcrumb=scrub_sentry_breadcrumb,
        )
        _configured = True
    except Exception:  # noqa: BLE001
        return


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
