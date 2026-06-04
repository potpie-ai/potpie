"""Event activity stream publisher (port).

Live-streaming surface for the events screen. Two kinds of streams:

- **per-event activity** — agent work events (tool calls, plans, errors)
  emitted as they happen. Keyed on ``event_id``. Subscribed to from the
  side-panel detail view.
- **per-pot status** — coarse status transitions (queued → processing →
  done / failed) for *all* events in a pot. Subscribed to from the list
  view so row indicators update without polling.

The port deliberately offers two flavours of subscription (``replay`` and
``replay_and_tail``) so HTTP handlers can express their needs explicitly:

- ``replay`` returns past events only and stops — useful for one-shot
  history fetches.
- ``replay_and_tail`` replays past then keeps yielding new events until the
  stream signals end or the consumer disconnects.

Adapters should be safe to call from sync code (Celery workers). The
replay/tail iterators are blocking generators; HTTP handlers run them in a
threadpool via ``StreamingResponse``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


class EventStreamPublisherPort(Protocol):
    """Publish + subscribe to live activity / status for events.

    Adapters fan ``publish_status`` to both the per-pot status stream and the
    per-event activity stream so a subscriber to either surface sees the
    transition. ``publish_activity`` carries one work event (tool call, plan,
    model_messages, etc.) — ``sequence`` is the agent's ordinal for this
    event_id so consumers can dedupe across replay/tail boundaries.

    Subscriber methods are blocking generators that yield dicts with at
    least ``stream_id`` and ``type`` keys; they terminate on an
    ``{"type": "end", ...}`` event or on idle timeout.
    """

    # ----- publish side -----

    def publish_status(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

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
    ) -> None: ...

    def publish_end(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        error: str | None = None,
    ) -> None: ...

    # ----- subscribe side -----

    def replay_and_tail_activity(
        self,
        *,
        event_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]: ...

    def replay_and_tail_pot_status(
        self,
        *,
        pot_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]: ...


class NoOpEventStreamPublisher:
    """Inert publisher. Used when streaming infra is unavailable.

    Implements the port shape with no side effects. Subscriptions yield a
    single ``end`` event so HTTP handlers don't hang.
    """

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
        del pot_id, event_id, status, stage, message, metadata

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
        del pot_id, event_id, run_id, kind, sequence, title, body, payload

    def publish_end(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        del pot_id, event_id, status, error

    def replay_and_tail_activity(
        self,
        *,
        event_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        del event_id, cursor, idle_timeout_seconds
        yield {
            "type": "end",
            "status": "disabled",
            "message": "Streaming is disabled on this server.",
            "stream_id": "0-0",
        }

    def replay_and_tail_pot_status(
        self,
        *,
        pot_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        del pot_id, cursor, idle_timeout_seconds
        yield {
            "type": "end",
            "status": "disabled",
            "message": "Streaming is disabled on this server.",
            "stream_id": "0-0",
        }


__all__ = ["EventStreamPublisherPort", "NoOpEventStreamPublisher"]
