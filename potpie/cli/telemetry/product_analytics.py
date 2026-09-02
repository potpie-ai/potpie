from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    """Spools the event; a detached flusher ships it (see :mod:`.spool`).

    The sink used to POST from the CLI process — first one event at a time on
    a fresh TLS connection each, then as one batch at exit — and either way
    the command's wall time carried the round trip. Now the process only
    writes a line. The timestamp travels with the event so the flusher's
    delay does not move it.
    """

    settings: ProductAnalyticsSettings

    def capture(self, event: ProductAnalyticsEvent) -> None:
        if not self.settings.enabled or self.settings.api_key is None:
            return
        payload: ProductAnalyticsPayload = {
            "event": event.name,
            "distinct_id": event.distinct_id,
            "properties": dict(event.properties),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _spool_product_analytics_event(payload)


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


def _batch_url(host: str) -> str:
    return f"{host.rstrip('/')}/batch/"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)


def _spool_product_analytics_event(payload: Mapping[str, object]) -> None:
    from . import spool

    spool.append({"kind": "analytics", "event": dict(payload)})


_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _http_client
    with _http_client_lock:
        if _http_client is None:
            _http_client = httpx.Client(timeout=_timeout(), follow_redirects=True)
        return _http_client


def _close_http_client() -> None:
    global _http_client
    with _http_client_lock:
        client, _http_client = _http_client, None
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - shutdown must never fail CLI work.
            return


def _post_product_analytics_batch(
    *,
    url: str,
    payload: Mapping[str, object],
) -> None:
    try:
        _http().post(url, json=payload)
    except Exception:  # noqa: BLE001 - product analytics must never affect CLI work.
        return


atexit.register(_close_http_client)


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
