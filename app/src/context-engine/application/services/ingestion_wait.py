"""Sync waiter for terminal ingestion state (same submission path as async; blocks on DB)."""

from __future__ import annotations

import time

from domain.ingestion_event_models import IngestionEvent
from domain.ports.ingestion_event_store import IngestionEventStore


def wait_for_terminal_ingestion_event(
    store: IngestionEventStore,
    event_id: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.5,
) -> IngestionEvent | None:
    """
    Poll until the event reaches ``done`` or ``error``, or ``timeout_seconds`` elapses.

    Returns the latest row snapshot (including non-terminal if timed out). Returns ``None``
    only if the event id does not exist on first read.
    """
    deadline = time.monotonic() + timeout_seconds
    last: IngestionEvent | None = None
    while time.monotonic() < deadline:
        ev = store.get_event(event_id)
        if ev is None:
            return None
        last = ev
        if ev.status in ("done", "error"):
            return ev
        time.sleep(max(poll_interval_seconds, 0.05))
    return last
