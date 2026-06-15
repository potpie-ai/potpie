from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, TypeAlias

import httpx

from .context import current_telemetry_context
from .settings import ProductAnalyticsSettings

AnalyticsValue: TypeAlias = str | int | float | bool | None | tuple[str, ...]
AnalyticsProperties: TypeAlias = Mapping[str, AnalyticsValue]


@dataclass(frozen=True, slots=True)
class ProductAnalyticsEvent:
    name: str
    distinct_id: str
    properties: AnalyticsProperties = field(default_factory=dict)


class ProductAnalyticsSink(Protocol):
    def capture(self, event: ProductAnalyticsEvent) -> None: ...


class NoOpProductAnalyticsSink:
    def capture(self, event: ProductAnalyticsEvent) -> None:
        del event


@dataclass(frozen=True, slots=True)
class PostHogSink:
    settings: ProductAnalyticsSettings

    def capture(self, event: ProductAnalyticsEvent) -> None:
        if not self.settings.enabled or self.settings.api_key is None:
            return
        payload = {
            "api_key": self.settings.api_key,
            "event": event.name,
            "distinct_id": event.distinct_id,
            "properties": dict(event.properties),
        }
        with httpx.Client(timeout=_timeout(), follow_redirects=True) as client:
            client.post(_capture_url(self.settings.host), json=payload)


_sink: ProductAnalyticsSink = NoOpProductAnalyticsSink()


def configure_product_analytics(settings: ProductAnalyticsSettings) -> None:
    global _sink
    if settings.enabled and settings.api_key is not None:
        _sink = PostHogSink(settings)
        return
    _sink = NoOpProductAnalyticsSink()


def set_product_analytics_sink(sink: ProductAnalyticsSink) -> None:
    global _sink
    _sink = sink


def capture_event(name: str, properties: AnalyticsProperties | None = None) -> None:
    telemetry = current_telemetry_context()
    if telemetry is None:
        return
    event_properties: dict[str, AnalyticsValue] = {
        **telemetry.analytics_properties(),
        **dict(properties or {}),
    }
    event = ProductAnalyticsEvent(
        name=name,
        distinct_id=telemetry.anonymous_install_id,
        properties=event_properties,
    )
    try:
        _sink.capture(event)
    except Exception:  # noqa: BLE001 - product analytics must never fail CLI work.
        return


def _capture_url(host: str) -> str:
    return f"{host.rstrip('/')}/capture/"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)


__all__ = [
    "AnalyticsProperties",
    "AnalyticsValue",
    "NoOpProductAnalyticsSink",
    "PostHogSink",
    "ProductAnalyticsEvent",
    "ProductAnalyticsSettings",
    "ProductAnalyticsSink",
    "capture_event",
    "configure_product_analytics",
    "set_product_analytics_sink",
]
