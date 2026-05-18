"""Per-runtime composition presets — the only place that knows the runtime.

Each returns an ObservabilityConfig; pass it to configure(). These encode the
audit's fixes so every entrypoint is correct by construction.

  monolith()    web app: console (dev) or json_stdout (prod), Sentry+logfire
                 enabled when DSN/token present, FastAPI middleware expected.
  celery()      workers: json_stdout, Sentry WITH CeleryIntegration, logfire
                 with instrument_pydantic_ai=False (OTel prefork bug),
                 backend init deferred to worker_process_init (EC2).
  standalone()  CLI / context-engine standalone: console, no Sentry/logfire
                 unless explicitly configured. Fixes the gap where standalone
                 had NO logging config and fell back to Python default
                 (root=WARNING, no JSON, no redaction).

EDGE CASE: profiles read env at call time (from_env) but callers may override
fields; precedence is explicit-arg > env > profile default.
"""

from __future__ import annotations

from .config import ObservabilityConfig


def monolith() -> ObservabilityConfig:
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2/3")


def celery() -> ObservabilityConfig:
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")


def standalone() -> ObservabilityConfig:
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")
