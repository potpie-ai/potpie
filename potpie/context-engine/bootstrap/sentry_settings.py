from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata
from typing import ClassVar, Final

_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class SentrySettings:
    __slots__: ClassVar[tuple[str, ...]] = (
        "dist",
        "dsn",
        "enabled",
        "environment",
        "release",
    )

    enabled: bool
    dsn: str | None
    environment: str
    release: str
    dist: str | None


def load_sentry_settings() -> SentrySettings:
    dsn = _env("POTPIE_SENTRY_DSN") or _env("SENTRY_DSN")
    telemetry_disabled = _is_truthy_flag("POTPIE_TELEMETRY_DISABLED")
    sentry_disabled = _is_falsey_flag("POTPIE_SENTRY_ENABLED")
    return SentrySettings(
        enabled=dsn is not None and not telemetry_disabled and not sentry_disabled,
        dsn=dsn,
        environment=telemetry_environment(),
        release=_env("POTPIE_SENTRY_RELEASE")
        or _env("SENTRY_RELEASE")
        or default_cli_release(),
        dist=_env("POTPIE_SENTRY_DIST") or _env("SENTRY_DIST"),
    )


def default_cli_release() -> str:
    try:
        version = metadata.version("potpie-context-engine")
    except metadata.PackageNotFoundError:
        version = "0.1.0"
    return f"potpie-cli@{version}"


def telemetry_environment() -> str:
    return _env("POTPIE_SENTRY_ENVIRONMENT") or _env("SENTRY_ENVIRONMENT") or "dev"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _is_falsey_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in _FALSE_VALUES


def _is_truthy_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized != "" and normalized not in _FALSE_VALUES
