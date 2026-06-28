from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from potpie.runtime.settings import load_runtime_settings
from potpie.runtime.telemetry.sentry_settings import (
    SentrySettings,
    default_cli_release,
    load_sentry_settings,
    telemetry_environment,
)


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


def load_product_analytics_settings() -> ProductAnalyticsSettings:
    settings = load_runtime_settings()
    return ProductAnalyticsSettings(
        enabled=(
            settings.posthog_api_key is not None
            and settings.product_analytics_enabled
            and not settings.telemetry_disabled
        ),
        api_key=settings.posthog_api_key,
        host=settings.posthog_host,
    )


__all__ = [
    "ProductAnalyticsSettings",
    "SentrySettings",
    "default_cli_release",
    "load_product_analytics_settings",
    "load_sentry_settings",
    "telemetry_environment",
]
