from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SentryConfig:
    enabled: bool = False
    dsn: str | None = None
    environment: str | None = None
    with_fastapi: bool = False
    with_celery: bool = False


@dataclass(slots=True)
class LogfireConfig:
    enabled: bool = False
    token: str | None = None
    instrument_pydantic_ai: bool = True


@dataclass(slots=True)
class ObservabilityConfig:
    service_name: str = "potpie"
    env: str = "development"
    level: str = "INFO"
    redact: bool = True
    sinks: list[str] = field(default_factory=lambda: ["console"])
    sentry: SentryConfig = field(default_factory=SentryConfig)
    logfire: LogfireConfig = field(default_factory=LogfireConfig)

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        import os

        env = (os.getenv("ENV") or "development").lower().strip()
        return cls(
            service_name=os.getenv("SERVICE_NAME", "potpie"),
            env=env,
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            sentry=SentryConfig(
                enabled=bool(os.getenv("SENTRY_DSN")),
                dsn=os.getenv("SENTRY_DSN"),
                environment=os.getenv("SENTRY_ENVIRONMENT", env),
            ),
            logfire=LogfireConfig(
                enabled=bool(os.getenv("LOGFIRE_TOKEN")),
                token=os.getenv("LOGFIRE_TOKEN"),
            ),
        )
