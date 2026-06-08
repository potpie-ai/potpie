"""Behavior tests for the event stream publisher port + adapters.

Covers:
- InMemory adapter recording publishes in order
- NoOp adapter yielding a single end on subscribe
- Redis adapter:
  - Status publish fans out to both pot + activity streams
  - Activity publish only hits the activity stream
  - replay+tail terminates on ``end`` events
  - replay+tail respects idle timeout when no entries arrive
  - Publish failures are swallowed (must not break ingestion)
"""

from __future__ import annotations

from typing import Any

import pytest

from adapters.outbound.event_stream.inmemory_publisher import (
    InMemoryEventStreamPublisher,
)
from adapters.outbound.event_stream.redis_publisher import (
    RedisEventStreamPublisher,
    _format_entry,
)
from domain.ports.event_stream import NoOpEventStreamPublisher


# ----- InMemory adapter -----


class TestInMemoryPublisher:
    def test_status_events_record_in_order(self) -> None:
        pub = InMemoryEventStreamPublisher()
        pub.publish_status(pot_id="p", event_id="e1", status="queued")
        pub.publish_status(pot_id="p", event_id="e2", status="processing")
        assert [e["event_id"] for e in pub.status_events] == ["e1", "e2"]
        assert pub.status_events[1]["status"] == "processing"

    def test_activity_events_capture_all_fields(self) -> None:
        pub = InMemoryEventStreamPublisher()
        pub.publish_activity(
            pot_id="p",
            event_id="e1",
            run_id="r1",
            kind="tool_call",
            sequence=1,
            title="context_search",
            body=None,
            payload={"args": {"query": "auth"}},
        )
        recorded = pub.activity_events[0]
        assert recorded["kind"] == "tool_call"
        assert recorded["title"] == "context_search"
        assert recorded["sequence"] == 1
        assert recorded["payload"] == {"args": {"query": "auth"}}

    def test_replay_only_returns_events_for_requested_event_id(self) -> None:
        pub = InMemoryEventStreamPublisher()
        pub.publish_activity(
            pot_id="p",
            event_id="e1",
            run_id=None,
            kind="plan",
            sequence=1,
        )
        pub.publish_activity(
            pot_id="p",
            event_id="e2",
            run_id=None,
            kind="plan",
            sequence=1,
        )
        pub.publish_end(pot_id="p", event_id="e1", status="done")
        out = list(pub.replay_and_tail_activity(event_id="e1"))
        kinds = [e["type"] for e in out]
        assert kinds == ["activity", "end"]
        # Last is the end marker for e1, not e2.
        assert out[-1]["status"] == "done"

    def test_pot_replay_filters_by_pot_id(self) -> None:
        pub = InMemoryEventStreamPublisher()
        pub.publish_status(pot_id="p1", event_id="e1", status="queued")
        pub.publish_status(pot_id="p2", event_id="e2", status="queued")
        pub.publish_status(pot_id="p1", event_id="e3", status="processing")
        out = list(pub.replay_and_tail_pot_status(pot_id="p1"))
        assert [e["event_id"] for e in out] == ["e1", "e3"]


# ----- NoOp adapter -----


class TestNoOpPublisher:
    def test_publish_is_silent_and_safe(self) -> None:
        pub = NoOpEventStreamPublisher()
        # None of these may raise.
        pub.publish_status(pot_id="p", event_id="e", status="queued")
        pub.publish_activity(
            pot_id="p", event_id="e", run_id=None, kind="plan", sequence=1
        )
        pub.publish_end(pot_id="p", event_id="e", status="done")

    def test_subscribe_yields_single_end_marker(self) -> None:
        pub = NoOpEventStreamPublisher()
        activity = list(pub.replay_and_tail_activity(event_id="e"))
        status = list(pub.replay_and_tail_pot_status(pot_id="p"))
        assert len(activity) == 1
        assert activity[0]["type"] == "end"
        assert activity[0]["status"] == "disabled"
        assert len(status) == 1
        assert status[0]["type"] == "end"


# ----- Redis adapter -----


class _FakeRedis:
    """Tiny in-memory Redis stub that supports the calls we make."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self.next_id = 1
        self.publish_raises = False
        self.xread_responses: list[Any] | None = None

    def xadd(
        self,
        key: str,
        fields: dict[str, Any],
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        if self.publish_raises:
            raise RuntimeError("redis down")
        entry_id = f"0-{self.next_id}"
        self.next_id += 1
        encoded = {
            (k.encode() if isinstance(k, str) else k): (
                v.encode() if isinstance(v, str) else v
            )
            for k, v in fields.items()
        }
        self.streams.setdefault(key, []).append((entry_id, encoded))
        return entry_id

    def expire(self, key: str, ttl: int) -> bool:
        return True

    def xrange(self, key: str, min: str = "-", max: str = "+") -> list:
        entries = self.streams.get(key, [])
        if min == "-":
            return list(entries)
        # Simple "after this id" semantics for tests
        return [e for e in entries if e[0] > min]

    def xread(self, streams: dict[str, str], block: int = 0, count: int = 10):
        if self.xread_responses is not None and self.xread_responses:
            return self.xread_responses.pop(0)
        return []


@pytest.fixture
def fake_redis_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RedisEventStreamPublisher, _FakeRedis]:
    fake = _FakeRedis()
    # Bypass the connect path — assign the fake directly.
    pub = RedisEventStreamPublisher.__new__(RedisEventStreamPublisher)
    pub._client = fake  # type: ignore[attr-defined]
    pub._ttl = 60
    pub._max_len = 100
    return pub, fake


class TestRedisPublisher:
    def test_publish_status_fans_to_both_streams(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        pub, fake = fake_redis_publisher
        pub.publish_status(pot_id="p1", event_id="e1", status="processing")
        assert "events:status:p1" in fake.streams
        assert "events:activity:e1" in fake.streams
        assert len(fake.streams["events:status:p1"]) == 1
        assert len(fake.streams["events:activity:e1"]) == 1

    def test_publish_activity_does_not_touch_pot_status_stream(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        pub, fake = fake_redis_publisher
        pub.publish_activity(
            pot_id="p1",
            event_id="e1",
            run_id=None,
            kind="plan",
            sequence=1,
        )
        assert "events:activity:e1" in fake.streams
        assert "events:status:p1" not in fake.streams

    def test_publish_failure_is_swallowed(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        # Liveness must not corrupt ingestion. Even with Redis down, publishes
        # complete without raising — caller has no way to recover anyway.
        pub, fake = fake_redis_publisher
        fake.publish_raises = True
        pub.publish_status(pot_id="p1", event_id="e1", status="processing")
        pub.publish_activity(
            pot_id="p1",
            event_id="e1",
            run_id=None,
            kind="plan",
            sequence=1,
        )
        # No streams created — but no exception raised either.
        assert fake.streams == {}

    def test_payload_json_round_trips(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        pub, fake = fake_redis_publisher
        pub.publish_activity(
            pot_id="p1",
            event_id="e1",
            run_id="r1",
            kind="tool_call",
            sequence=3,
            payload={"args": {"query": "auth", "limit": 5}},
        )
        out = list(
            pub.replay_and_tail_activity(event_id="e1", idle_timeout_seconds=0.05)
        )
        # First event is the activity; second is the idle_timeout end marker.
        activity = next(e for e in out if e.get("type") == "activity")
        assert activity["kind"] == "tool_call"
        assert activity["sequence"] == 3
        assert activity["payload"] == {"args": {"query": "auth", "limit": 5}}

    def test_replay_stops_on_end_event(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        pub, fake = fake_redis_publisher
        pub.publish_activity(
            pot_id="p1",
            event_id="e1",
            run_id=None,
            kind="plan",
            sequence=1,
        )
        pub.publish_end(pot_id="p1", event_id="e1", status="done")
        # publish_activity afterwards should NOT be yielded since end ended the loop.
        pub.publish_activity(
            pot_id="p1",
            event_id="e1",
            run_id=None,
            kind="plan",
            sequence=2,
        )
        out = list(pub.replay_and_tail_activity(event_id="e1"))
        # Activity, end, (no more — replay stops on end). The third activity is
        # past the end marker in the stream and would never be yielded.
        assert out[-1]["type"] == "end"
        assert out[-1]["status"] == "done"

    def test_replay_emits_idle_timeout_when_no_activity(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        pub, fake = fake_redis_publisher
        # No publishes — should hit idle timeout almost immediately.
        out = list(
            pub.replay_and_tail_activity(event_id="e1", idle_timeout_seconds=0.01)
        )
        assert out
        assert out[-1]["type"] == "end"
        assert out[-1]["status"] == "idle_timeout"

    def test_pot_stream_does_not_stop_on_end_event(
        self, fake_redis_publisher: tuple[RedisEventStreamPublisher, _FakeRedis]
    ) -> None:
        # A pot stream tracks many events; one event ending must not close
        # the connection. Idle timeout closes it instead.
        pub, fake = fake_redis_publisher
        pub.publish_status(pot_id="p1", event_id="e1", status="done")
        pub.publish_end(pot_id="p1", event_id="e1", status="done")
        out = list(
            pub.replay_and_tail_pot_status(
                pot_id="p1",
                idle_timeout_seconds=0.01,
            )
        )
        # We see the status entry, the end-for-event-e1 entry, then idle timeout.
        kinds = [e.get("type") for e in out]
        # End-for-event-e1 IS yielded but is_not the terminator for the pot stream.
        assert "status" in kinds
        # The trailing entry is the idle_timeout end, NOT the per-event end.
        assert out[-1]["type"] == "end"
        assert out[-1]["status"] == "idle_timeout"


class TestFormatEntry:
    def test_decodes_bytes_fields(self) -> None:
        entry = _format_entry(
            b"1736942000000-0",
            {b"type": b"activity", b"kind": b"plan", b"sequence": b"3"},
        )
        assert entry["stream_id"] == "1736942000000-0"
        assert entry["type"] == "activity"
        assert entry["sequence"] == 3

    def test_parses_json_suffix_fields(self) -> None:
        entry = _format_entry(
            "0-1",
            {"type": "activity", "payload_json": '{"x": 1}'},
        )
        assert entry["payload"] == {"x": 1}
        # The suffixed key is stripped — consumers see the natural name.
        assert "payload_json" not in entry

    def test_invalid_json_suffix_becomes_none(self) -> None:
        entry = _format_entry("0-1", {"payload_json": "not valid"})
        assert entry["payload"] is None
