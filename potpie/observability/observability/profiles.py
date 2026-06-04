"""Per-runtime composition presets — the only place that knows the runtime.

Each returns an ObservabilityConfig; pass it to configure().

  monolith()    web app: from_env (console in dev / json_stdout otherwise;
                 Sentry/logfire auto-enabled only when creds present).
  standalone()  CLI / context-engine standalone: console + redaction, no
                 Sentry/logfire. Fixes the audit gap where standalone had NO
                 logging config and fell back to Python default (root=WARNING,
                 no JSON, no redaction).
  celery()      Phase 3 — needs Sentry CeleryIntegration + worker_process_init
                 backend init + instrument_pydantic_ai=False.
"""

from __future__ import annotations

import os

from .config import LogfireConfig, ObservabilityConfig, SentryConfig


def monolith() -> ObservabilityConfig:
    cfg = ObservabilityConfig.from_env()
    cfg.sentry.with_fastapi = True
    return cfg


def standalone() -> ObservabilityConfig:
    return ObservabilityConfig(
        env=(os.getenv("ENV") or "development").lower().strip(),
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        sinks=["console"],
        redact=True,
        sentry=SentryConfig(enabled=False),
        logfire=LogfireConfig(enabled=False),
    )


def celery() -> ObservabilityConfig:
    """Workers: JSONL out, Sentry with CeleryIntegration, logfire WITHOUT
    pydantic-ai instrumentation (OTel prefork contextvar bug). Backend init
    happens inside the worker process via integrations.celery (EC2).
    """
    cfg = ObservabilityConfig.from_env()
    cfg.sinks = ["json_stdout"]
    if cfg.sentry.enabled:
        cfg.sentry.with_celery = True
    cfg.logfire.instrument_pydantic_ai = False
    return cfg
