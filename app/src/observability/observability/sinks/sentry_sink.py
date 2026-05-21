"""Sentry sink (extra: observability[sentry]) — NEW; fixes the audit's core gap.

Today (pre-package) Sentry is web-only, prod-only, DSN-often-unset, and there
is NO loguru->Sentry bridge so logger.error/exception calls never reach
Sentry. This sink closes that:

 - setup(): sentry_sdk.init with the config; idempotent per-process. Network
   init is acceptable here because integrations/celery.py calls configure()
   INSIDE worker_process_init (EC2) — never at module import for workers.
   default_integrations=False + explicit list (preserves the audit's
   Strawberry-not-installed safety from the original setup_sentry).
 - build_handler(): handler at config.sentry.event_level (default ERROR)
   that calls capture_exception/capture_message, attaching obs_context +
   obs_fields as scope tags / extras.
 - own_logging_integration=True (default): include LoggingIntegration with
   event_level=None so THIS sink owns the log -> event path (no double
   capture).
 - Disabled-state contract: enabled=False or no DSN -> setup() emits ONE
   visible warning and build_handler returns None (no silent no-op).
"""

from __future__ import annotations

import logging
import os

from ..config import ObservabilityConfig

_HINT = "sentry sink requires sentry-sdk — install observability[sentry]"
_LEVEL_TO_SENTRY = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "fatal",
}
_logger = logging.getLogger(__name__)
_state = {"initialised_pid": None}


class _SentryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            import sentry_sdk

            fields: dict = {}
            for attr in ("obs_context", "obs_fields"):
                data = getattr(record, attr, None)
                if isinstance(data, dict):
                    fields.update(data)
            # new_scope is the sentry-sdk 2.x preferred API (push_scope deprecated).
            with sentry_sdk.new_scope() as scope:
                for k, v in fields.items():
                    try:
                        scope.set_tag(str(k), str(v))
                    except Exception:
                        pass
                scope.set_extra("logger", record.name)
                scope.set_extra("function", record.funcName)
                scope.set_extra("line", record.lineno)
                if record.exc_info:
                    sentry_sdk.capture_exception(record.exc_info)
                else:
                    level = _LEVEL_TO_SENTRY.get(record.levelname, "info")
                    sentry_sdk.capture_message(record.getMessage(), level=level)
        except Exception:
            self.handleError(record)


class SentrySink:
    name = "sentry"

    def setup(self, config: ObservabilityConfig) -> None:
        sc = config.sentry
        if not sc.enabled or not sc.dsn:
            _logger.warning(
                "sentry disabled: %s",
                "enabled=False" if not sc.enabled else "no SENTRY_DSN",
            )
            return
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_HINT) from exc

        if _state["initialised_pid"] == os.getpid():
            return  # idempotent within a process

        integrations: list = []
        if sc.own_logging_integration:
            # event_level=None disables Sentry's auto log->event capture;
            # our _SentryHandler owns that path.
            integrations.append(LoggingIntegration(event_level=None))
        if sc.with_fastapi:
            try:
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                integrations.append(FastApiIntegration())
            except Exception as exc:
                _logger.debug("FastApiIntegration skipped: %s", exc)
        if sc.with_celery:
            try:
                from sentry_sdk.integrations.celery import CeleryIntegration
                integrations.append(CeleryIntegration())
            except Exception as exc:
                _logger.debug("CeleryIntegration skipped: %s", exc)

        try:
            sentry_sdk.init(
                dsn=sc.dsn,
                environment=sc.environment,
                traces_sample_rate=sc.traces_sample_rate,
                profiles_sample_rate=sc.profiles_sample_rate,
                default_integrations=False,
                integrations=integrations,
            )
            _state["initialised_pid"] = os.getpid()
        except Exception as exc:
            _logger.warning("sentry_sdk.init failed: %s", exc)

    def build_handler(
        self, config: ObservabilityConfig
    ) -> logging.Handler | None:
        sc = config.sentry
        if not sc.enabled or not sc.dsn:
            return None
        handler = _SentryHandler()
        handler.setLevel(sc.event_level)
        return handler

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
