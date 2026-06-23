from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Final

from context_engine.bootstrap.sentry_settings import (
    SentrySettings,
    default_cli_release,
    load_sentry_settings,
    telemetry_environment,
)

_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


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
    api_key = _env("POTPIE_POSTHOG_API_KEY")
    telemetry_disabled = _is_truthy_flag("POTPIE_TELEMETRY_DISABLED")
    product_disabled = _is_falsey_flag("POTPIE_POSTHOG_ENABLED") or _is_falsey_flag(
        "POTPIE_PRODUCT_ANALYTICS_ENABLED"
    )
    return ProductAnalyticsSettings(
        enabled=api_key is not None and not telemetry_disabled and not product_disabled,
        api_key=api_key,
        host=_env("POTPIE_POSTHOG_HOST") or "https://us.i.posthog.com",
    )


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


__all__ = [
    "ProductAnalyticsSettings",
    "SentrySettings",
    "default_cli_release",
    "load_product_analytics_settings",
    "load_sentry_settings",
    "telemetry_environment",
]
