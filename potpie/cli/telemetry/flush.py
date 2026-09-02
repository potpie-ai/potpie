"""The detached telemetry flusher: ``python -m potpie.cli.telemetry.flush``.

Started by a CLI process at exit (see :mod:`potpie.cli.telemetry.spool`).
Takes the spool, ships analytics as batch POSTs and metrics through the Sentry
SDK, and repeats while commands keep appending. Bounded rounds, one flusher
at a time, exit 0 no matter what: nobody is waiting for this process, and a
failure here is a failure to ship telemetry, which the next command's exit
retries with whatever spooled since.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable

MAX_ROUNDS = 5
ANALYTICS_BATCH_SIZE = 50
METRIC_KINDS = frozenset({"count", "distribution", "gauge"})


def main() -> int:
    from . import spool

    if not spool.acquire_lock():
        return 0
    try:
        for _ in range(MAX_ROUNDS):
            records = spool.take()
            if not records:
                break
            ship(records)
    except Exception:  # noqa: BLE001 - nothing waits on this process.
        pass
    finally:
        spool.release_lock()
    return 0


def ship(records: list[dict[str, Any]]) -> None:
    analytics = [r for r in records if r.get("kind") == "analytics"]
    metrics = [r for r in records if r.get("kind") == "metric"]
    if analytics:
        try:
            ship_analytics(analytics)
        except Exception:  # noqa: BLE001
            pass
    if metrics:
        try:
            ship_metrics(metrics)
        except Exception:  # noqa: BLE001
            pass


def ship_analytics(records: list[dict[str, Any]]) -> None:
    from .product_analytics import (
        _batch_url,
        _close_http_client,
        _post_product_analytics_batch,
    )
    from .settings import load_product_analytics_settings

    settings = load_product_analytics_settings()
    if not settings.enabled or settings.api_key is None:
        return
    events = [r["event"] for r in records if isinstance(r.get("event"), dict)]
    try:
        for chunk in _chunks(events, ANALYTICS_BATCH_SIZE):
            _post_product_analytics_batch(
                url=_batch_url(settings.host),
                payload={"api_key": settings.api_key, "batch": chunk},
            )
    finally:
        _close_http_client()


def ship_metrics(records: list[dict[str, Any]]) -> None:
    from potpie_context_engine.bootstrap import sentry_metrics_runtime as runtime

    from .settings import load_sentry_settings

    settings = load_sentry_settings()
    if not settings.enabled:
        return
    runtime.configure_metrics(settings, short_lived_process=True)
    if not runtime.metrics_configured():
        return
    for record in records:
        kind = record.get("type")
        if kind in METRIC_KINDS:
            _emit_metric(getattr(runtime, str(kind)), record)
    runtime.flush(timeout=5.0)


def _emit_metric(emit: Any, record: dict[str, Any]) -> None:
    try:
        emit(
            str(record["name"]),
            record.get("value", 1),
            unit=record.get("unit"),
            attributes=record.get("attributes") or None,
        )
    # One malformed record must not cost the rest of the batch.
    except Exception:  # noqa: BLE001
        return


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


if __name__ == "__main__":
    sys.exit(main())
