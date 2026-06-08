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
           that captures logged exceptions, attaching obs_context +
          obs_fields as scope tags / extras.
         - own_logging_integration=True (default): include LoggingIntegration while
           disabling automatic log events and breadcrumbs.
         - Disabled-state contract: enabled=False or no DSN -> no network handler.
"""

from __future__ import annotations

import logging
import os

from ..config import ObservabilityConfig
from .sentry_privacy import (
    SentryRateLimiter,
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
    split_sentry_fields,
)

_HINT = "sentry sink requires sentry-sdk — install observability[sentry]"
_LEVEL_TO_SENTRY = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "fatal",
}
_logger = logging.getLogger(__name__)
_state = {"client_key": None}
_disabled_notices: set[tuple[int, str]] = set()


class _SentryHandler(logging.Handler):
    def __init__(self, config: ObservabilityConfig) -> None:
        super().__init__()
        self._config = config

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import sentry_sdk

            fields: dict = {
                "service": self._config.service_name,
                "environment": self._config.sentry.environment or self._config.env,
                "release": self._config.sentry.release,
            }
            for attr in ("obs_context", "obs_fields"):
                data = getattr(record, attr, None)
                if isinstance(data, dict):
                    fields.update(data)
            if not record.exc_info:
                fields.setdefault("error.code", "logged_error")
                fields.setdefault("error.kind", "unexpected")
                fields.setdefault("is_expected", "false")
            tags, contexts = split_sentry_fields(fields)
            # new_scope is the sentry-sdk 2.x preferred API (push_scope deprecated).
            with sentry_sdk.new_scope() as scope:
                for k, v in tags.items():
                    try:
                        scope.set_tag(k, v)
                    except Exception as exc:
                        _logger.debug("sentry tag skipped: %s", exc)
                for name, context in contexts.items():
                    scope.set_context(name, context)
                scope.set_extra("logger", record.name)
                scope.set_extra("function", record.funcName)
                scope.set_extra("line", record.lineno)
                if record.exc_info:
                    sentry_sdk.capture_exception(record.exc_info)
                else:
                    error_code = tags.get("error.code", "logged_error")
                    level = _LEVEL_TO_SENTRY.get(record.levelname, "error")
                    sentry_sdk.capture_message(f"potpie.{error_code}", level=level)
        except Exception as exc:
            _logger.debug("sentry capture failed: %s", exc)


class SentrySink:
    name = "sentry"

    def setup(self, config: ObservabilityConfig) -> None:
        sc = config.sentry
        if not sc.enabled or not sc.dsn:
            reason = "enabled=False" if not sc.enabled else "no SENTRY_DSN"
            notice_key = (os.getpid(), reason)
            if notice_key not in _disabled_notices:
                _logger.warning("sentry disabled: %s", reason)
                _disabled_notices.add(notice_key)
            return
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_HINT) from exc

        client_key = (os.getpid(), sc.dsn, sc.environment, sc.release, id(sc.transport))
        if _state["client_key"] == client_key:
            return  # idempotent within a process

        integrations: list = []
        if sc.own_logging_integration:
            # event_level=None disables Sentry's auto log->event capture;
            # level=None disables automatic log breadcrumbs as a privacy default.
            integrations.append(
                LoggingIntegration(
                    level=None,
                    event_level=None,
                    sentry_logs_level=None,
                )
            )
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

        rate_limiter = SentryRateLimiter(sc.max_events_per_minute)

        def before_send(event, hint):
            if sc.before_send is not None:
                event = sc.before_send(event, hint)
                if event is None:
                    return None
            return scrub_sentry_event(event, hint, rate_limiter=rate_limiter)

        def before_breadcrumb(breadcrumb, hint):
            if sc.before_breadcrumb is not None:
                breadcrumb = sc.before_breadcrumb(breadcrumb, hint)
                if breadcrumb is None:
                    return None
            return scrub_sentry_breadcrumb(breadcrumb, hint)

        try:
            sentry_sdk.init(
                dsn=sc.dsn,
                environment=sc.environment,
                release=sc.release,
                dist=sc.dist,
                send_default_pii=sc.send_default_pii,
                before_send=before_send,
                before_breadcrumb=before_breadcrumb,
                transport=sc.transport,
                include_source_context=sc.include_source_context,
                traces_sample_rate=sc.traces_sample_rate,
                profiles_sample_rate=sc.profiles_sample_rate,
                default_integrations=False,
                integrations=integrations,
            )
            _state["client_key"] = client_key
        except Exception as exc:
            _logger.debug("sentry_sdk.init failed: %s", exc)

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        sc = config.sentry
        if not sc.enabled or not sc.dsn or not sc.capture_error_logs:
            return None
        handler = _SentryHandler(config)
        handler.setLevel(sc.event_level)
        return handler

    def instrument(self, config: ObservabilityConfig) -> None:
        return None

    def shutdown(self, config: ObservabilityConfig) -> None:
        # Flush pending events on reconfigure / process exit. Bounded timeout
        # so a slow Sentry endpoint can't hang shutdown.
        try:
            import sentry_sdk

            try:
                sentry_sdk.flush(timeout=2.0)
            except Exception as exc:
                _logger.debug("sentry flush failed: %s", exc)
        except Exception as exc:
            _logger.debug("sentry shutdown skipped: %s", exc)
