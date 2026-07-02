from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata
from typing import ClassVar

from potpie.runtime.env import env_value
from potpie.runtime.settings import (
    RuntimeSettings,
    build_git_sha,
    load_runtime_settings,
)


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
    return sentry_settings_from_runtime(load_runtime_settings())


def sentry_settings_from_runtime(settings: RuntimeSettings) -> SentrySettings:
    return SentrySettings(
        enabled=(
            settings.sentry_dsn is not None
            and settings.sentry_enabled
            and not settings.telemetry_disabled
        ),
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=_env("POTPIE_SENTRY_RELEASE")
        or _env("SENTRY_RELEASE")
        or default_cli_release(),
        dist=_env("POTPIE_SENTRY_DIST") or _env("SENTRY_DIST") or build_git_sha(),
    )


def default_cli_release() -> str:
    try:
        version = metadata.version("potpie")
    except metadata.PackageNotFoundError:
        try:
            version = metadata.version("potpie-context-engine")
        except metadata.PackageNotFoundError:
            version = "0.1.0"
    return f"potpie-cli@{version}"


def telemetry_environment() -> str:
    return load_runtime_settings().environment


def _env(name: str) -> str | None:
    return env_value(os.environ, name)


__all__ = [
    "SentrySettings",
    "default_cli_release",
    "load_sentry_settings",
    "sentry_settings_from_runtime",
    "telemetry_environment",
]
