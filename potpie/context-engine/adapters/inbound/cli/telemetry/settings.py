from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from adapters.inbound.cli.telemetry import _build_defaults as build_defaults
from adapters.inbound.cli.telemetry.preferences import telemetry_enabled_by_preference
from bootstrap.sentry_settings import (
    SentrySettings,
    default_cli_release,
)

_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
TelemetryState = Literal["blocked", "disabled", "enabled"]
TelemetrySinkStatus = Literal["anonymous", "blocked", "disabled"]


@dataclass(frozen=True)
class TelemetryStatus:
    __slots__: ClassVar[tuple[str, ...]] = (
        "telemetry",
        "crash_reports",
        "analytics",
    )

    telemetry: TelemetryState
    crash_reports: TelemetrySinkStatus
    analytics: TelemetrySinkStatus


@dataclass(frozen=True)
class ProductAnalyticsSettings:
    __slots__: ClassVar[tuple[str, ...]] = (
        "api_key",
        "enabled",
        "host",
    )

    enabled: bool
    api_key: str | None
    host: str


def load_sentry_settings() -> SentrySettings:
    dsn = (
        _env("POTPIE_SENTRY_DSN")
        or _env("SENTRY_DSN")
        or _baked(build_defaults.POTPIE_SENTRY_DSN)
    )
    telemetry_disabled = _telemetry_globally_disabled()
    sentry_disabled = _is_falsey_config(
        "POTPIE_SENTRY_ENABLED",
        build_defaults.POTPIE_SENTRY_ENABLED,
    )
    user_enabled = telemetry_enabled_by_preference()
    return SentrySettings(
        enabled=dsn is not None
        and user_enabled
        and not telemetry_disabled
        and not sentry_disabled,
        dsn=dsn,
        environment=telemetry_environment(),
        release=_env("POTPIE_SENTRY_RELEASE")
        or _env("SENTRY_RELEASE")
        or _baked(build_defaults.POTPIE_SENTRY_RELEASE)
        or default_cli_release(),
        dist=(
            _env("POTPIE_SENTRY_DIST")
            or _env("SENTRY_DIST")
            or _baked(build_defaults.POTPIE_SENTRY_DIST)
        ),
    )


def load_product_analytics_settings() -> ProductAnalyticsSettings:
    api_key = _env("POTPIE_POSTHOG_API_KEY") or _baked(
        build_defaults.POTPIE_POSTHOG_API_KEY
    )
    telemetry_disabled = _telemetry_globally_disabled()
    product_disabled = _is_falsey_config(
        "POTPIE_POSTHOG_ENABLED",
        build_defaults.POTPIE_POSTHOG_ENABLED,
    ) or _is_falsey_config(
        "POTPIE_PRODUCT_ANALYTICS_ENABLED",
        build_defaults.POTPIE_PRODUCT_ANALYTICS_ENABLED,
    )
    user_enabled = telemetry_enabled_by_preference()
    return ProductAnalyticsSettings(
        enabled=api_key is not None
        and user_enabled
        and not telemetry_disabled
        and not product_disabled,
        api_key=api_key,
        host=(
            _env("POTPIE_POSTHOG_HOST")
            or _baked(build_defaults.POTPIE_POSTHOG_HOST)
            or "https://us.i.posthog.com"
        ),
    )


def telemetry_environment() -> str:
    return (
        _env("POTPIE_SENTRY_ENVIRONMENT")
        or _env("SENTRY_ENVIRONMENT")
        or _baked(build_defaults.POTPIE_SENTRY_ENVIRONMENT)
        or build_defaults.DEFAULT_POTPIE_SENTRY_ENVIRONMENT
    )


def load_telemetry_status() -> TelemetryStatus:
    sentry_settings = load_sentry_settings()
    product_settings = load_product_analytics_settings()
    if _telemetry_globally_disabled():
        telemetry: TelemetryState = "blocked"
    elif not telemetry_enabled_by_preference():
        telemetry = "disabled"
    else:
        telemetry = "enabled"
    return TelemetryStatus(
        telemetry=telemetry,
        crash_reports=_sink_status(sentry_settings.enabled, telemetry),
        analytics=_sink_status(product_settings.enabled, telemetry),
    )


def _sink_status(
    enabled: bool, telemetry: TelemetryState
) -> TelemetrySinkStatus:
    if telemetry == "blocked":
        return "blocked"
    if telemetry == "disabled":
        return "disabled"
    if enabled:
        return "anonymous"
    return "disabled"


def _telemetry_globally_disabled() -> bool:
    return _is_truthy_config(
        "POTPIE_TELEMETRY_DISABLED",
        build_defaults.POTPIE_TELEMETRY_DISABLED,
    )


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _baked(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _flag_config(name: str, baked: str) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value.strip().lower()
    baked_value = _baked(baked)
    if baked_value is None:
        return None
    return baked_value.lower()


def _is_falsey_config(name: str, baked: str = "") -> bool:
    value = _flag_config(name, baked)
    return value is not None and value in _FALSE_VALUES


def _is_truthy_config(name: str, baked: str = "") -> bool:
    value = _flag_config(name, baked)
    return value is not None and value != "" and value not in _FALSE_VALUES


__all__ = [
    "ProductAnalyticsSettings",
    "SentrySettings",
    "TelemetryStatus",
    "default_cli_release",
    "load_product_analytics_settings",
    "load_sentry_settings",
    "load_telemetry_status",
    "telemetry_environment",
]
