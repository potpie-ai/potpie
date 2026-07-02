from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Literal

from potpie.runtime.env import clean_env_value, env_value, flag_value
from potpie.runtime.settings import RuntimeSettings
from potpie.runtime.telemetry.preferences import (
    TelemetryState,
    load_runtime_settings_with_telemetry_preference,
)
from potpie.runtime.telemetry.sentry_settings import (
    SentrySettings,
    default_cli_release,
    sentry_settings_from_runtime,
    telemetry_environment,
)

TelemetrySinkStatus = Literal["anonymous", "blocked", "disabled"]
_CODE_DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"


@dataclass(frozen=True)
class TelemetryResolution:
    __slots__: ClassVar[tuple[str, ...]] = (
        "runtime",
        "telemetry",
    )

    runtime: RuntimeSettings
    telemetry: TelemetryState


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


def load_telemetry_resolution() -> TelemetryResolution:
    resolution = load_runtime_settings_with_telemetry_preference()
    return TelemetryResolution(
        runtime=resolution.runtime,
        telemetry=resolution.telemetry,
    )


def load_cli_runtime_settings() -> RuntimeSettings:
    return load_telemetry_resolution().runtime


def load_sentry_settings() -> SentrySettings:
    return sentry_settings_from_runtime(load_cli_runtime_settings())


def load_product_analytics_settings() -> ProductAnalyticsSettings:
    return _product_analytics_settings_from_runtime(load_cli_runtime_settings())


def load_telemetry_status() -> TelemetryStatus:
    resolution = load_telemetry_resolution()
    sentry_settings = sentry_settings_from_runtime(resolution.runtime)
    product_settings = _product_analytics_settings_from_runtime(resolution.runtime)
    return TelemetryStatus(
        telemetry=resolution.telemetry,
        crash_reports=_sink_status(sentry_settings.enabled, resolution.telemetry),
        analytics=_sink_status(product_settings.enabled, resolution.telemetry),
    )


def _product_analytics_settings_from_runtime(
    settings: RuntimeSettings,
) -> ProductAnalyticsSettings:
    defaults = _load_product_analytics_defaults()
    posthog_enabled = _flag(
        _env("POTPIE_POSTHOG_ENABLED") or defaults.get("posthog_enabled") or "1"
    )
    product_analytics_enabled = _flag(
        _env("POTPIE_PRODUCT_ANALYTICS_ENABLED")
        or defaults.get("product_analytics_enabled")
        or "1"
    )
    api_key = _env("POTPIE_POSTHOG_API_KEY") or defaults.get("posthog_api_key")
    host = (
        _env("POTPIE_POSTHOG_HOST")
        or defaults.get("posthog_host")
        or _CODE_DEFAULT_POSTHOG_HOST
    ).rstrip("/")
    return ProductAnalyticsSettings(
        enabled=(
            api_key is not None
            and posthog_enabled
            and product_analytics_enabled
            and not settings.telemetry_disabled
        ),
        api_key=api_key,
        host=host,
    )


def _sink_status(enabled: bool, telemetry: TelemetryState) -> TelemetrySinkStatus:
    if telemetry == "blocked":
        return "blocked"
    if telemetry == "disabled":
        return "disabled"
    if enabled:
        return "anonymous"
    return "disabled"


def _load_product_analytics_defaults() -> dict[str, str]:
    defaults: dict[str, str] = {}
    try:
        from potpie.cli.telemetry import _build_config as posthog_defaults
    except ImportError:
        return defaults
    _copy_constant(
        defaults,
        "posthog_enabled",
        posthog_defaults,
        "POTPIE_POSTHOG_ENABLED",
    )
    _copy_constant(
        defaults,
        "product_analytics_enabled",
        posthog_defaults,
        "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    )
    _copy_constant(
        defaults,
        "posthog_api_key",
        posthog_defaults,
        "POTPIE_POSTHOG_API_KEY",
    )
    _copy_constant(defaults, "posthog_host", posthog_defaults, "POTPIE_POSTHOG_HOST")
    return defaults


def _copy_constant(
    target: dict[str, str],
    key: str,
    source: object,
    name: str,
) -> None:
    value = _clean(getattr(source, name, None))
    if value is not None:
        target[key] = value


def _env(name: str) -> str | None:
    return env_value(os.environ, name)


def _clean(value: object) -> str | None:
    return clean_env_value(value)


def _flag(value: str) -> bool:
    return flag_value(value)


__all__ = [
    "ProductAnalyticsSettings",
    "SentrySettings",
    "TelemetryResolution",
    "TelemetryStatus",
    "default_cli_release",
    "load_cli_runtime_settings",
    "load_product_analytics_settings",
    "load_sentry_settings",
    "load_telemetry_resolution",
    "load_telemetry_status",
    "telemetry_environment",
]
