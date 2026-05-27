"""Composition root for observability inside context-engine.

context-engine is treated as an independent module — when its CLI / MCP /
HTTP entry point boots, it owns observability setup itself. When the
monolith hosts context-engine code, the monolith has already called
`observability.configure(...)` and we skip (PID check) to avoid clobbering
its sinks.

Backend: logfire only. No Loki, no Sentry — that's a deliberate scope
choice for this module.
"""

from __future__ import annotations

import logging

from observability import get_logger
import os

logger = get_logger(__name__)


def configure_observability() -> None:
    """Initialise the observability package for the current process.

    Idempotent and host-aware:
    - If the monolith (or any other host) already called configure() in this
      process, this is a no-op.
    - If logfire is not installed or initialisation fails, we log a warning
      and continue — context-engine never refuses to boot because tracing
      isn't available.
    """
    try:
        from observability import _state as _obs_state
        from observability import configure
        from observability.config import (
            LogfireConfig,
            ObservabilityConfig,
            SentryConfig,
            parse_bool,
        )
    except ModuleNotFoundError:
        # Observability is a hard dep, so this should never fire — but if a
        # standalone install skipped the extras, don't crash the entry point.
        logger.warning(
            "observability package not importable; logs will use stdlib defaults"
        )
        return

    if _obs_state.get("configured_pid") == os.getpid():
        # The host (monolith / test harness) has already configured this
        # process. Don't replace its sinks.
        return

    logfire_token = os.getenv("LOGFIRE_TOKEN") or None
    # Production: logfire only. Dev / unconfigured: fall back to console so
    # devs still see their own logs locally. (logfire sink returns no
    # handler when the token is absent, which would leave root silent.)
    sinks = ["logfire"] if logfire_token else ["console"]
    cfg = ObservabilityConfig(
        service_name=os.getenv("SERVICE_NAME", "context-engine"),
        env=(os.getenv("ENV") or "development").lower().strip(),
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        redact=True,
        sinks=sinks,
        sentry=SentryConfig(enabled=False),
        logfire=LogfireConfig(
            enabled=bool(logfire_token),
            token=logfire_token,
            send_to_cloud=parse_bool(
                os.getenv("LOGFIRE_SEND_TO_CLOUD"),
                default=True,
            ),
            project_name=os.getenv("LOGFIRE_PROJECT_NAME", "context-engine"),
            # Defensive default for forked workers (Hatchet etc.). If/when a
            # caller proves the OTel contextvar issue doesn't apply, flip
            # this via env or override at the call site.
            instrument_pydantic_ai=False,
        ),
    )

    try:
        configure(cfg)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("observability.configure failed: %s", exc, exc=exc)
