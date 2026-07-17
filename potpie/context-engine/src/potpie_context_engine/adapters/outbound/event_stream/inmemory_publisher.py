"""In-memory :class:`EventStreamPublisherPort` for unit tests.

Captures published events in lists so tests can assert the publisher was
called with the right shape. Replay yields collected events in order, then
a single ``end`` event so iterators close cleanly.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InMemoryEventStreamPublisher:
    """Thread-safe in-memory recorder for tests.

    The dataclass holds two lists:

    - ``status_events``: every ``publish_status`` call.
    - ``activity_events``: every ``publish_activity`` call.

    Both lists preserve call order so a test can assert sequencing.
    """

    status_events: list[dict[str, Any]] = field(default_factory=list)
    activity_events: list[dict[str, Any]] = field(default_factory=list)
    end_events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def publish_status(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.status_events.append(
                {
                    "pot_id": pot_id,
                    "event_id": event_id,
                    "status": status,
                    "stage": stage,
                    "message": message,
                    "metadata": metadata,
                }
            )

    def publish_activity(
        self,
        *,
        pot_id: str,
        event_id: str,
        run_id: str | None,
        kind: str,
        sequence: int,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.activity_events.append(
                {
                    "pot_id": pot_id,
                    "event_id": event_id,
                    "run_id": run_id,
                    "kind": kind,
                    "sequence": sequence,
                    "title": title,
                    "body": body,
                    "payload": payload,
                }
            )

    def publish_end(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.end_events.append(
                {
                    "pot_id": pot_id,
                    "event_id": event_id,
                    "status": status,
                    "error": error,
                }
            )

    def replay_and_tail_activity(
        self,
        *,
        event_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        del cursor, idle_timeout_seconds
        with self._lock:
            events = list(self.activity_events)
            end = next(
                (e for e in self.end_events if e.get("event_id") == event_id),
                None,
            )
        for idx, ev in enumerate(events):
            if ev.get("event_id") != event_id:
                continue
            yield {
                "stream_id": f"0-{idx + 1}",
                "type": "activity",
                **{k: v for k, v in ev.items() if v is not None},
            }
        if end is not None:
            yield {
                "stream_id": "end",
                "type": "end",
                "status": end["status"],
                "error": end.get("error"),
            }

    def replay_and_tail_pot_status(
        self,
        *,
        pot_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        del cursor, idle_timeout_seconds
        with self._lock:
            events = [e for e in self.status_events if e.get("pot_id") == pot_id]
        for idx, ev in enumerate(events):
            yield {
                "stream_id": f"0-{idx + 1}",
                "type": "status",
                **{k: v for k, v in ev.items() if v is not None},
            }


__all__ = ["InMemoryEventStreamPublisher"]
