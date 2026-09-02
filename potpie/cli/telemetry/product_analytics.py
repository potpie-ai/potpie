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
#: How long exit may wait for the batch in flight. A CLI command runs for
#: ~0.3 s; analytics that cannot leave within this bound are dropped rather
#: than paid for by the caller's wall time.
_DISPATCH_WORKER_JOIN_TIMEOUT_SECONDS: Final[float] = 1.5
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


class _QueuedProductAnalyticsEvent:
    __slots__ = ("api_key", "payload", "url")

    def __init__(
        self, *, url: str, api_key: str, payload: Mapping[str, object]
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.payload = payload


class _ProductAnalyticsDispatcher:
    """Ships every event a process captured as one batch POST.

    The previous shape opened a fresh ``httpx.Client`` — a new TCP and TLS
    handshake — per event and posted them one at a time on a non-daemon
    thread, so ``search`` (three events) held the process for ~2.4 s after its
    answer was already printed. Events now queue, the worker drains whatever is
    queued into a single ``/batch/`` request over one keep-alive client, and
    the thread is a daemon: ``flush`` at exit gives it a bounded window, after
    which the interpreter leaves without it.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[_QueuedProductAnalyticsEvent] = queue.Queue(
            maxsize=_DISPATCH_QUEUE_MAX_SIZE
        )
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def dispatch(
        self, *, url: str, api_key: str, payload: Mapping[str, object]
    ) -> None:
        try:
            self._queue.put_nowait(
                _QueuedProductAnalyticsEvent(url=url, api_key=api_key, payload=payload)
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
                daemon=True,
                name="potpie-product-analytics",
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=_DISPATCH_WORKER_IDLE_TIMEOUT_SECONDS)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            batch = [first]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                for (url, api_key), events in _group_by_destination(batch).items():
                    _post_product_analytics_batch(
                        url=url,
                        payload={
                            "api_key": api_key,
                            "batch": [dict(event.payload) for event in events],
                        },
                    )
            finally:
                for _ in batch:
                    self._queue.task_done()


def _group_by_destination(
    events: list[_QueuedProductAnalyticsEvent],
) -> dict[tuple[str, str], list[_QueuedProductAnalyticsEvent]]:
    grouped: dict[tuple[str, str], list[_QueuedProductAnalyticsEvent]] = {}
    for event in events:
        grouped.setdefault((event.url, event.api_key), []).append(event)
    return grouped


@dataclass(frozen=True, slots=True)
class PostHogSink:
    settings: ProductAnalyticsSettings

    def capture(self, event: ProductAnalyticsEvent) -> None:
        if not self.settings.enabled or self.settings.api_key is None:
            return
        payload: ProductAnalyticsPayload = {
            "event": event.name,
            "distinct_id": event.distinct_id,
            "properties": dict(event.properties),
        }
        _send_product_analytics_event(
            _batch_url(self.settings.host),
            api_key=self.settings.api_key,
            payload=payload,
        )


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


def _batch_url(host: str) -> str:
    return f"{host.rstrip('/')}/batch/"


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)


def _send_product_analytics_event(
    url: str,
    *,
    api_key: str,
    payload: Mapping[str, object],
) -> None:
    _dispatcher.dispatch(url=url, api_key=api_key, payload=payload)


def _flush_product_analytics_dispatcher() -> None:
    _dispatcher.flush()


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


def _shutdown_product_analytics() -> None:
    _flush_product_analytics_dispatcher()
    _close_http_client()


atexit.register(_shutdown_product_analytics)


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
