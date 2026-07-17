from __future__ import annotations

import atexit
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Final, Mapping, Protocol, TypeAlias

import httpx

from .context import current_telemetry_context
from .settings import ProductAnalyticsSettings

AnalyticsValue: TypeAlias = str | int | float | bool | None | tuple[str, ...]
AnalyticsProperties: TypeAlias = Mapping[str, AnalyticsValue]
ProductAnalyticsPayload: TypeAlias = dict[str, str | AnalyticsProperties]
_DISPATCH_QUEUE_MAX_SIZE: Final[int] = 128
_DISPATCH_WORKER_IDLE_TIMEOUT_SECONDS: Final[float] = 0.1
_DISPATCH_WORKER_JOIN_TIMEOUT_SECONDS: Final[float] = 5.0
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


class _QueuedProductAnalyticsPayload:
    __slots__ = ("payload", "url")

    def __init__(self, *, url: str, payload: Mapping[str, object]) -> None:
        self.url = url
        self.payload = payload


class _ProductAnalyticsDispatcher:
    def __init__(self) -> None:
        self._queue: queue.Queue[_QueuedProductAnalyticsPayload] = queue.Queue(
            maxsize=_DISPATCH_QUEUE_MAX_SIZE
        )
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def dispatch(self, *, url: str, payload: Mapping[str, object]) -> None:
        try:
            self._queue.put_nowait(
                _QueuedProductAnalyticsPayload(url=url, payload=payload)
            )
        except queue.Full:
            return
        self._ensure_worker()

    def flush(self) -> None:
        deadline = time.monotonic() + _DISPATCH_WORKER_JOIN_TIMEOUT_SECONDS
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run,
                daemon=False,
                name="potpie-product-analytics",
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                queued_payload = self._queue.get(
                    timeout=_DISPATCH_WORKER_IDLE_TIMEOUT_SECONDS
                )
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                _post_product_analytics_payload(
                    url=queued_payload.url,
                    payload=queued_payload.payload,
                )
            finally:
                self._queue.task_done()


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
_dispatcher = _ProductAnalyticsDispatcher()


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
    payload: Mapping[str, object],
) -> None:
    _dispatcher.dispatch(url=url, payload=payload)


def _flush_product_analytics_dispatcher() -> None:
    _dispatcher.flush()


def _post_product_analytics_payload(
    *,
    url: str,
    payload: Mapping[str, object],
) -> None:
    try:
        with httpx.Client(timeout=_timeout(), follow_redirects=True) as client:
            client.post(url, json=payload)
    except Exception:  # noqa: BLE001 - product analytics must never affect CLI work.
        return


atexit.register(_flush_product_analytics_dispatcher)


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
