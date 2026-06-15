from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final, Mapping, Protocol, TypeAlias

import httpx

from .context import current_telemetry_context
from .settings import ProductAnalyticsSettings

AnalyticsValue: TypeAlias = str | int | float | bool | None | tuple[str, ...]
AnalyticsProperties: TypeAlias = Mapping[str, AnalyticsValue]
ProductAnalyticsPayload: TypeAlias = dict[str, str | AnalyticsProperties]
_CANONICAL_ANALYTICS_PROPERTY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "anonymous_install_id",
        "invocation_id",
        "daemon_session_id",
        "environment",
        "output_mode",
        "cli_version",
        "python_version",
        "platform",
        "arch",
    }
)


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
        payload: ProductAnalyticsPayload = {
            "api_key": self.settings.api_key,
            "event": event.name,
            "distinct_id": event.distinct_id,
            "properties": dict(event.properties),
        }
        _send_product_analytics_payload(_capture_url(self.settings.host), payload)


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
    telemetry_properties = telemetry.analytics_properties()
    event_properties: dict[str, AnalyticsValue] = {
        **telemetry_properties,
        **dict(properties or {}),
    }
    event_properties.update(
        {
            key: telemetry_properties[key]
            for key in _CANONICAL_ANALYTICS_PROPERTY_KEYS
            if key in telemetry_properties
        }
    )
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


def _send_product_analytics_payload(
    url: str,
    payload: ProductAnalyticsPayload,
) -> None:
    thread = threading.Thread(
        target=_post_product_analytics_payload,
        kwargs={"url": url, "payload": payload},
        daemon=True,
        name="potpie-product-analytics",
    )
    thread.start()


def _post_product_analytics_payload(
    *,
    url: str,
    payload: ProductAnalyticsPayload,
) -> None:
    try:
        with httpx.Client(timeout=_timeout(), follow_redirects=True) as client:
            client.post(url, json=payload)
    except Exception:  # noqa: BLE001 - product analytics must never affect CLI work.
        return


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
