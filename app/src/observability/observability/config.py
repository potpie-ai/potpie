"""Observability configuration model (pure data — fully defined contract).

Config is an explicit object passed into `configure()`. Env-loading is ONE
optional adapter (`from_env`), so the package never assumes a host's env
conventions. Env var names below are the ones the audit found in the current
codebase, carried forward for migration parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SentryConfig:
    """Sentry (error capture). GAP the audit found: today Sentry is prod-only,
    web-only, no Celery, DSN often unset -> silent no-op. Contract fix:
    `enabled` is derived (explicit flag AND dsn present); configure() emits ONE
    visible 'sentry disabled: no DSN' log instead of silently doing nothing."""

    enabled: bool = False
    dsn: str | None = None
    environment: str | None = None
    traces_sample_rate: float = 0.25
    # Audit flagged profiles_sample_rate=1.0 in prod as a cost smell.
    profiles_sample_rate: float = 0.05
    # EC: if a Sentry log-sink is registered, Sentry's own LoggingIntegration
    # must be set event_level=None to avoid double-capture.
    own_logging_integration: bool = True


@dataclass(slots=True)
class LogfireConfig:
    """logfire (tracing + optional log export). Carries the known Celery
    escape hatch: instrument_pydantic_ai must be False in prefork workers
    ('Token was created in a different Context' OTel contextvar bug)."""

    enabled: bool = False
    token: str | None = None
    send_to_cloud: bool = True
    project_name: str = "potpie"
    instrument_litellm: bool = True
    instrument_pydantic_ai: bool = True  # set False in Celery profile


@dataclass(slots=True)
class ObservabilityConfig:
    """Top-level config. `sinks` are sink names resolved via the registry, or
    pre-built Sink instances. Default per-library levels mirror the audit's
    existing dial-down map (uvicorn/sqlalchemy/httpx/etc.)."""

    service_name: str = "potpie"
    env: str = "development"
    level: str = "INFO"
    redact: bool = True
    show_stack_traces: bool = True
    sinks: list[str] = field(default_factory=lambda: ["console"])
    library_levels: dict[str, str] = field(default_factory=dict)
    sentry: SentryConfig = field(default_factory=SentryConfig)
    logfire: LogfireConfig = field(default_factory=LogfireConfig)

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Build from env vars: ENV, LOG_LEVEL, LOG_STACK_TRACES, SENTRY_DSN,
        LOGFIRE_TOKEN, LOGFIRE_SEND_TO_CLOUD, LOGFIRE_PROJECT_NAME, CELERY_WORKER.

        STUB (Phase 1): contract only.
        """
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")
