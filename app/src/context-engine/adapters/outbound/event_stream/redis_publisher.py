"""Redis Streams adapter for :class:`EventStreamPublisherPort`.

Two stream key schemes:

- ``events:activity:{event_id}`` — per-event work-event timeline. Replayed
  in full when a user opens the side-panel detail; then tailed live.
- ``events:status:{pot_id}`` — coarse lifecycle deltas for every event in a
  pot. Tailed by the list view so row indicators update without polling.

Mirrors the chat ``RedisStreamManager`` so operators have one mental model
for stream TTL / max-len / wait behaviour. We don't import the existing
class so the context-engine package stays self-contained — instead we use
the same key conventions and call the same Redis primitives.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

try:
    import redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - redis is an extras dep
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Defaults match the chat stream manager so operators tune one knob.
_DEFAULT_STREAM_TTL_SECONDS = 60 * 60  # 1 hour
_DEFAULT_STREAM_MAX_LEN = 10_000


class RedisEventStreamPublisher:
    """Redis-backed publisher + subscriber for event activity / status.

    Constructed against a Redis URL (defaults to ``REDIS_URL`` env). When
    Redis is unreachable, publish methods log and swallow — they must never
    block ingestion. Subscriber methods raise so HTTP handlers surface the
    error to the client.
    """

    ACTIVITY_KEY_FMT = "events:activity:{event_id}"
    STATUS_KEY_FMT = "events:status:{pot_id}"

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        stream_ttl_seconds: int | None = None,
        stream_max_len: int | None = None,
    ) -> None:
        if redis is None:
            raise RuntimeError(
                "redis package not installed; cannot build RedisEventStreamPublisher"
            )
        url = redis_url or os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError(
                "REDIS_URL not configured; cannot build RedisEventStreamPublisher"
            )
        self._client = redis.from_url(
            url,
            socket_connect_timeout=10,
            socket_timeout=30,
            decode_responses=False,
        )
        self._ttl = stream_ttl_seconds or _DEFAULT_STREAM_TTL_SECONDS
        self._max_len = stream_max_len or _DEFAULT_STREAM_MAX_LEN

    # ----- key helpers -----

    @classmethod
    def activity_key(cls, event_id: str) -> str:
        return cls.ACTIVITY_KEY_FMT.format(event_id=event_id)

    @classmethod
    def status_key(cls, pot_id: str) -> str:
        return cls.STATUS_KEY_FMT.format(pot_id=pot_id)

    # ----- publish -----

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
        payload = {
            "type": "status",
            "event_id": event_id,
            "pot_id": pot_id,
            "status": status,
        }
        if stage is not None:
            payload["stage"] = stage
        if message is not None:
            payload["message"] = message
        if metadata:
            payload["metadata_json"] = json.dumps(metadata, default=_safe_json)
        # Fan to both surfaces so either subscriber sees the transition.
        self._safe_publish(self.status_key(pot_id), payload)
        self._safe_publish(self.activity_key(event_id), payload)

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
        data: dict[str, Any] = {
            "type": "activity",
            "event_id": event_id,
            "pot_id": pot_id,
            "kind": kind,
            "sequence": str(sequence),
        }
        if run_id is not None:
            data["run_id"] = run_id
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if payload:
            # _json suffix mirrors the chat stream convention so the consumer
            # auto-parses it back to a structured value.
            data["payload_json"] = json.dumps(payload, default=_safe_json)
        self._safe_publish(self.activity_key(event_id), data)

    def publish_end(
        self,
        *,
        pot_id: str,
        event_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        payload = {
            "type": "end",
            "event_id": event_id,
            "pot_id": pot_id,
            "status": status,
        }
        if error is not None:
            payload["error"] = error
        # Fan to both: per-event consumers can stop, list consumers see the
        # transition and update the row.
        self._safe_publish(self.activity_key(event_id), payload)
        self._safe_publish(self.status_key(pot_id), payload)

    # ----- subscribe -----

    def replay_and_tail_activity(
        self,
        *,
        event_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        key = self.activity_key(event_id)
        yield from self._replay_and_tail(
            key=key, cursor=cursor, idle_timeout_seconds=idle_timeout_seconds
        )

    def replay_and_tail_pot_status(
        self,
        *,
        pot_id: str,
        cursor: str | None = None,
        idle_timeout_seconds: float = 30.0,
    ) -> Iterator[dict[str, Any]]:
        key = self.status_key(pot_id)
        yield from self._replay_and_tail(
            key=key,
            cursor=cursor,
            idle_timeout_seconds=idle_timeout_seconds,
            # Pot streams never naturally "end" — they cover every event in
            # the pot — so the consumer disconnects when the user navigates
            # away. We still cap by idle timeout to avoid wedged connections.
            stop_on_end=False,
        )

    # ----- internals -----

    def _safe_publish(self, key: str, data: dict[str, Any]) -> None:
        """XADD with bounded retention; swallow + log on Redis failures.

        Ingestion correctness must not depend on the stream — a Redis
        outage may degrade liveness UX but must not fail the agent run.
        """
        try:
            entry = {
                **{
                    k: v if isinstance(v, (str, bytes)) else str(v)
                    for k, v in data.items()
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.xadd(  # pyright: ignore[reportArgumentType]
                key, entry, maxlen=self._max_len, approximate=True
            )
            self._client.expire(key, self._ttl)
        except Exception as exc:  # noqa: BLE001 - liveness must not fail ingestion
            logger.warning("event stream publish to %s failed: %s", key, exc)

    def _replay_and_tail(
        self,
        *,
        key: str,
        cursor: str | None,
        idle_timeout_seconds: float,
        stop_on_end: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Replay past entries (from ``cursor`` or full), then live-tail.

        On a fresh subscription (no cursor) we replay full history so the
        side-panel scrubs in immediately on open. Reconnecting clients pass
        the last ``stream_id`` they saw to avoid double-rendering.
        """
        # 1) Replay past entries.
        try:
            min_id = cursor or "-"
            past = self._client.xrange(key, min=min_id, max="+")  # pyright: ignore[reportGeneralTypeIssues]
        except Exception as exc:
            logger.error("event stream replay (%s) failed: %s", key, exc)
            yield _end_event(status="error", message=f"replay failed: {exc}")
            return

        last_id = "0-0"
        for entry_id, fields in past:
            event = _format_entry(entry_id, fields)
            last_id = event["stream_id"]
            yield event
            if stop_on_end and event.get("type") == "end":
                return

        # 2) Tail new entries until end or idle timeout.
        deadline = time.monotonic() + idle_timeout_seconds
        while True:
            try:
                # Block up to 5s waiting for new entries. We loop so the
                # idle-timeout / cancellation check stays responsive.
                entries = self._client.xread(  # pyright: ignore[reportGeneralTypeIssues]
                    {key: last_id}, block=5000, count=10
                )
            except Exception as exc:
                logger.error("event stream tail (%s) failed: %s", key, exc)
                yield _end_event(status="error", message=f"tail failed: {exc}")
                return

            if not entries:
                if time.monotonic() >= deadline:
                    yield _end_event(
                        status="idle_timeout",
                        message="No activity within idle window.",
                        stream_id=last_id,
                    )
                    return
                continue

            # Reset idle clock — we got something.
            deadline = time.monotonic() + idle_timeout_seconds

            for _stream_key, stream_entries in entries:  # noqa: B007 - key unused
                for entry_id, fields in stream_entries:
                    event = _format_entry(entry_id, fields)
                    last_id = event["stream_id"]
                    yield event
                    if stop_on_end and event.get("type") == "end":
                        return


def _format_entry(entry_id: Any, fields: dict[Any, Any]) -> dict[str, Any]:
    """Render a Redis stream entry into a JSON-shaped dict.

    Fields ending ``_json`` are parsed back to their structured value with
    the suffix dropped — matches the chat stream convention.
    """
    out: dict[str, Any] = {
        "stream_id": entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id),
    }
    for k, v in fields.items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        value = v.decode() if isinstance(v, bytes) else v
        if key.endswith("_json") and isinstance(value, str):
            try:
                out[key[:-5]] = json.loads(value)
            except Exception:
                logger.warning("event stream: invalid JSON in field %s", key)
                out[key[:-5]] = None
        elif key == "sequence" and isinstance(value, str):
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
        else:
            out[key] = value
    return out


def _end_event(
    *,
    status: str,
    message: str | None = None,
    stream_id: str = "0-0",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "end",
        "status": status,
        "stream_id": stream_id,
    }
    if message is not None:
        out["message"] = message
    return out


def _safe_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = ["RedisEventStreamPublisher"]
