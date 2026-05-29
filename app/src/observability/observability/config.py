"""Observability configuration model (pure data — fully defined contract).

Config is an explicit object passed into `configure()`. Env-loading is ONE
optional adapter (`from_env`), so the package never assumes a host's env
conventions. Env var names below are the ones the audit found in the current
codebase, carried forward for migration parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_TRUTHY = {"1", "true", "yes", "y"}
_FALSY = {"0", "false", "no", "n"}


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.lower().strip()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default


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
    # Which integrations to add. Audit-faithful: default_integrations=False +
    # explicit list (preserves Strawberry-not-installed safety from the
    # original setup_sentry).
    with_fastapi: bool = False
    with_celery: bool = False
    # Sink handler emits events at this level and above.
    event_level: str = "ERROR"


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
        SENTRY_ENVIRONMENT, LOGFIRE_TOKEN, LOGFIRE_SEND_TO_CLOUD,
        LOGFIRE_PROJECT_NAME, CELERY_WORKER.

        Enablement is DERIVED, not silent: Sentry/logfire are only enabled when
        their credential is actually present (the audit's silent-no-op fix).
        """
        import os

        env = (os.getenv("ENV") or "development").lower().strip()
        is_dev = env == "development"
        sentry_dsn = os.getenv("SENTRY_DSN") or None
        logfire_token = os.getenv("LOGFIRE_TOKEN") or None

        return cls(
            service_name=os.getenv("SERVICE_NAME", "potpie"),
            env=env,
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            redact=True,
            show_stack_traces=os.getenv("LOG_STACK_TRACES", "true").lower()
            in ("true", "1", "yes"),
            sinks=["console"] if is_dev else ["json_stdout"],
            sentry=SentryConfig(
                enabled=bool(sentry_dsn),
                dsn=sentry_dsn,
                environment=os.getenv("SENTRY_ENVIRONMENT", env),
            ),
            logfire=LogfireConfig(
                enabled=bool(logfire_token),
                token=logfire_token,
                send_to_cloud=parse_bool(
                    os.getenv("LOGFIRE_SEND_TO_CLOUD"),
                    default=True,
                ),
                project_name=os.getenv("LOGFIRE_PROJECT_NAME", "potpie"),
                # OTel prefork bug: never instrument pydantic-ai in a worker.
                instrument_pydantic_ai=os.getenv("CELERY_WORKER") != "1",
            ),
        )
