"""Sentry sink (extra: observability[sentry]) — NEW; fixes the audit's core gap.

Today Sentry is web-only, prod-only, DSN-often-unset, and there is NO
loguru->Sentry bridge so 473 logger.error + 318 logger.exception calls never
reach Sentry. This sink closes that:

 - setup(): sentry_sdk.init with the config (idempotent, fork-safe — actual
   init for Celery happens in integrations/celery.py worker_process_init,
   EC2). If config.sentry.own_logging_integration: init Sentry's
   LoggingIntegration with event_level=None so THIS sink owns the log path
   (no double capture).
 - build_handler(): a handler at ERROR+ that calls capture_exception/
   capture_message, attaching log_context() correlation IDs as Sentry tags.
 - GAP: do NOT pipe INFO/volume logs to Sentry — errors only. Sentry is not a
   log aggregator.
 - Disabled-state contract: if not enabled / no DSN, setup() emits ONE
   visible warning and the sink becomes a no-op (no silent no-op).
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig


class SentrySink:
    name = "sentry"

    def setup(self, config: ObservabilityConfig) -> None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
