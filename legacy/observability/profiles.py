from __future__ import annotations

import os

from .config import LogfireConfig, ObservabilityConfig, SentryConfig


def monolith() -> ObservabilityConfig:
    cfg = ObservabilityConfig.from_env()
    cfg.sentry.with_fastapi = True
    return cfg


def standalone() -> ObservabilityConfig:
    return ObservabilityConfig(
        service_name=os.getenv("SERVICE_NAME", "potpie"),
        env=(os.getenv("ENV") or "development").lower().strip(),
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        sinks=["console"],
        sentry=SentryConfig(enabled=False),
        logfire=LogfireConfig(enabled=False),
    )


def celery() -> ObservabilityConfig:
    cfg = ObservabilityConfig.from_env()
    cfg.sinks = ["console"]
    cfg.sentry.with_celery = cfg.sentry.enabled
    cfg.logfire.instrument_pydantic_ai = False
    return cfg
