from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Literal

from adapters.inbound.cli.telemetry.preferences import telemetry_enabled_by_preference
from bootstrap.runtime_settings import RuntimeSettings, load_runtime_settings
from bootstrap.sentry_settings import (
    SentrySettings,
    default_cli_release,
    sentry_settings_from_runtime,
    telemetry_environment,
)

TelemetryState = Literal["blocked", "disabled", "enabled"]
TelemetrySinkStatus = Literal["anonymous", "blocked", "disabled"]


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
    runtime = load_runtime_settings()
    if runtime.telemetry_disabled:
        return TelemetryResolution(runtime=runtime, telemetry="blocked")
    if telemetry_enabled_by_preference():
        return TelemetryResolution(runtime=runtime, telemetry="enabled")
    return TelemetryResolution(
        runtime=replace(runtime, telemetry_disabled=True),
        telemetry="disabled",
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
    return ProductAnalyticsSettings(
        enabled=(
            settings.posthog_api_key is not None
            and settings.product_analytics_enabled
            and not settings.telemetry_disabled
        ),
        api_key=settings.posthog_api_key,
        host=settings.posthog_host,
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
