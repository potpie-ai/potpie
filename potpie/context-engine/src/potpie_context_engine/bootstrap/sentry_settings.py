"""Engine-owned Sentry metrics settings for standalone composition.

Potpie hosts supply the same structural settings from their own product runtime
configuration.  The loader here exists only for the independently runnable
Context Engine HTTP surface and deliberately knows nothing about Potpie auth,
analytics, UI, daemon, or other product configuration.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import ClassVar

from potpie_context_engine.bootstrap import standalone_env


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


def load_sentry_settings(
    environ: Mapping[str, str] | None = None,
    *,
    distribution_defaults: Mapping[str, object] | None = None,
) -> SentrySettings:
    """Load standalone engine metrics settings without importing a host package."""
    defaults = _normalize(
        load_distribution_defaults()
        if distribution_defaults is None
        else distribution_defaults
    )
    if environ is None:
        environment = (
            _env(os.environ, "POTPIE_ENVIRONMENT")
            or defaults.get("environment")
            or "dev"
        )
        if environment == "dev":
            standalone_env.load_standalone_env()
        source = os.environ
    else:
        source = environ
    dsn = _env(source, "POTPIE_SENTRY_DSN") or defaults.get("sentry_dsn")
    telemetry_disabled = _flag(_env(source, "POTPIE_TELEMETRY_DISABLED") or "0")
    sentry_enabled = _flag(_env(source, "POTPIE_SENTRY_ENABLED") or "1")
    return SentrySettings(
        enabled=dsn is not None and sentry_enabled and not telemetry_disabled,
        dsn=dsn,
        environment=(
            _env(source, "POTPIE_ENVIRONMENT") or defaults.get("environment") or "dev"
        ),
        release=(
            _env(source, "POTPIE_SENTRY_RELEASE") or f"potpie-cli@{_engine_version()}"
        ),
        dist=_env(source, "POTPIE_SENTRY_DIST") or build_git_sha(),
    )


def load_distribution_defaults() -> Mapping[str, str]:
    try:
        from potpie_context_engine.bootstrap._distribution_defaults import (
            DISTRIBUTION_DEFAULTS,
        )
    except ImportError:
        return MappingProxyType({})
    return MappingProxyType(_normalize(DISTRIBUTION_DEFAULTS))


def build_git_sha() -> str | None:
    try:
        from potpie_context_engine.bootstrap._build_info import GIT_SHA
    except ImportError:
        return None
    return _clean(GIT_SHA)


def _engine_version() -> str:
    try:
        return metadata.version("potpie-context-engine")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _normalize(values: Mapping[str, object]) -> dict[str, str]:
    return {
        str(key): cleaned
        for key, value in values.items()
        if (cleaned := _clean(value)) is not None
    }


def _env(environ: Mapping[str, str], name: str) -> str | None:
    return _clean(environ.get(name))


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


__all__ = ["SentrySettings", "load_sentry_settings"]
