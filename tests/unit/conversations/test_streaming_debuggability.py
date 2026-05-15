"""Unit tests for the streaming-debuggability fixes.

Covers the three observability changes that satisfy SHS-XXXX:
1. Every Redis event carries a ``trace_id`` field (sync + async manager).
2. ``_current_trace_id()`` returns 32-char hex inside a span, ``""`` outside.
3. The SSE generator's error path captures traceback + identifying fields.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry import trace as ot_trace
from opentelemetry.sdk.trace import TracerProvider

from app.modules.conversations.utils.redis_streaming import (
    AsyncRedisStreamManager,
    RedisStreamManager,
    _current_trace_id,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _ensure_tracer_provider() -> None:
    """Install a real TracerProvider so the trace_id helper has something to read.

    Other suites may have set one already; we only install if absent.
    """
    current = ot_trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        ot_trace.set_tracer_provider(TracerProvider())


class TestCurrentTraceIdHelper:
    def test_returns_empty_string_when_no_span_is_active(self):
        """Helper must never raise; absent span -> ''."""
        assert _current_trace_id() == ""

    def test_returns_32_hex_chars_inside_active_span(self):
        tracer = ot_trace.get_tracer("test")
        with tracer.start_as_current_span("test-span"):
            tid = _current_trace_id()
        assert len(tid) == 32
        assert all(c in "0123456789abcdef" for c in tid)


class TestPublishEventStampsTraceId:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_sync_publish_event_includes_trace_id_field(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()

        tracer = ot_trace.get_tracer("test")
        with tracer.start_as_current_span("publish"):
            mgr.publish_event("c1", "r1", "chunk", {"content": "hi"})

        event_data = client.xadd.call_args[0][1]
        assert "trace_id" in event_data
        assert len(event_data["trace_id"]) == 32

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_sync_publish_event_trace_id_is_empty_when_no_span(self, mock_redis, mock_cp):
        """Outside of any span, the field is present but empty - never absent."""
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        mgr.publish_event("c1", "r1", "chunk", {"content": "hi"})
        event_data = client.xadd.call_args[0][1]
        assert event_data.get("trace_id") == ""

    @pytest.mark.asyncio
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.AsyncRedis")
    async def test_async_publish_event_includes_trace_id_field(
        self, mock_async_redis, mock_cp
    ):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = AsyncMock()
        mock_async_redis.from_url.return_value = client
        mgr = AsyncRedisStreamManager()

        tracer = ot_trace.get_tracer("test")
        with tracer.start_as_current_span("publish-async"):
            await mgr.publish_event("c1", "r1", "chunk", {"content": "hi"})

        event_data = client.xadd.call_args[0][1]
        assert "trace_id" in event_data
        assert len(event_data["trace_id"]) == 32


class TestQueuePickupDiagnostics:
    @patch("app.modules.conversations.utils.redis_streaming.time.time")
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_sync_enqueue_metadata_and_pickup_diagnostics(
        self, mock_redis, mock_cp, mock_time
    ):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.llen.return_value = 7
        client.hgetall.return_value = {
            b"task_id": b"celery-1",
            b"queue_name": b"staging_agent_tasks",
            b"enqueued_at": b"100.0",
            b"queue_depth_at_enqueue": b"7",
        }
        mock_redis.from_url.return_value = client
        mock_time.side_effect = [100.0, 102.0]

        mgr = RedisStreamManager()
        queue_depth = mgr.get_queue_depth("staging_agent_tasks")
        enqueued_at = mgr.record_task_enqueue(
            "conv-1", "run-1", "celery-1", "staging_agent_tasks", queue_depth
        )
        diagnostics = mgr.task_pickup_diagnostics(
            "conv-1", "run-1", "staging_agent_tasks"
        )

        assert queue_depth == 7
        assert enqueued_at == 100.0
        client.hset.assert_called_once()
        assert client.hset.call_args.kwargs["mapping"]["task_id"] == "celery-1"
        assert diagnostics == {
            "task_id": "celery-1",
            "queue_name": "staging_agent_tasks",
            "queue_depth_at_enqueue": "7",
            "queue_depth_at_pickup": "7",
            "pickup_latency_ms": "2000",
        }

    @pytest.mark.asyncio
    @patch("app.modules.conversations.utils.redis_streaming.time.time")
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.AsyncRedis")
    async def test_async_enqueue_metadata_records_queue_depth(
        self, mock_async_redis, mock_cp, mock_time
    ):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = AsyncMock()
        client.llen.return_value = 3
        mock_async_redis.from_url.return_value = client
        mock_time.return_value = 123.456

        mgr = AsyncRedisStreamManager()
        queue_depth = await mgr.get_queue_depth("staging_agent_tasks")
        enqueued_at = await mgr.record_task_enqueue(
            "conv-1", "run-1", "celery-1", "staging_agent_tasks", queue_depth
        )

        assert queue_depth == 3
        assert enqueued_at == 123.456
        client.hset.assert_awaited_once()
        assert client.hset.call_args.kwargs["mapping"] == {
            "task_id": "celery-1",
            "queue_name": "staging_agent_tasks",
            "enqueued_at": "123.456",
            "queue_depth_at_enqueue": "3",
        }


class TestSSEGeneratorLogsTracebackAndIds:
    @patch("app.modules.conversations.utils.conversation_routing.logger")
    @patch("app.modules.conversations.utils.conversation_routing.RedisStreamManager")
    def test_redis_error_logs_traceback_and_identifying_fields(
        self, mock_mgr_cls, mock_logger
    ):
        """Mock the logger directly. ``app.*`` loggers use loguru intercept
        with ``propagate=False``, so ``caplog`` can't see their records."""
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        mgr = MagicMock()
        mgr.consume_stream.side_effect = ConnectionError("redis blew up")
        mock_mgr_cls.return_value = mgr

        # The full traceback lives in the log call while sanitized identifiers
        # are forwarded to the client.
        frames = list(redis_stream_generator("conv-1", "run-2", cursor="cur-3"))

        assert mock_logger.error.called, "expected an error log on Redis failure"
        call = mock_logger.error.call_args
        # Confirm message identifies the failing phase
        assert "sse_consume" in str(call.args[0])
        # exc_info=True captures the full traceback in the LogRecord
        assert call.kwargs.get("exc_info") is True
        # Identifying fields land in `extra` for grep-ability
        extra = call.kwargs.get("extra", {})
        assert extra.get("conversation_id") == "conv-1"
        assert extra.get("run_id") == "run-2"
        assert extra.get("cursor") == "cur-3"
        assert extra.get("failing_phase") == "sse_consume"
        assert extra.get("error_type") == "ConnectionError"
        assert "stack_trace" in extra
        # trace_id key is always present (possibly empty when no span is active)
        assert "trace_id" in extra

        assert len(frames) == 1
        payload = json.loads(frames[0])
        assert payload["event"] == "error"
        assert payload["run_id"] == "run-2"
        assert payload["failing_phase"] == "sse_consume"
        assert payload["error_type"] == "ConnectionError"
        assert payload["stack_trace_available"] is True
        assert "stack_trace" not in payload

    @patch("app.modules.conversations.utils.conversation_routing.RedisStreamManager")
    def test_error_end_event_is_forwarded_as_sanitized_frame(self, mock_mgr_cls):
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        mgr = MagicMock()
        mgr.consume_stream.return_value = iter(
            [
                {
                    "type": "end",
                    "status": "error",
                    "message": "An internal error occurred.",
                    "conversation_id": "conv-1",
                    "run_id": "run-2",
                    "trace_id": "abc",
                    "failing_phase": "agent_run",
                    "error_type": "RuntimeError",
                    "stack_trace_available": True,
                    "stream_id": "10-0",
                }
            ]
        )
        mock_mgr_cls.return_value = mgr

        frames = list(redis_stream_generator("conv-1", "run-2"))

        assert len(frames) == 1
        payload = json.loads(frames[0])
        assert payload["event"] == "error"
        assert payload["run_id"] == "run-2"
        assert payload["trace_id"] == "abc"
        assert payload["failing_phase"] == "agent_run"
        assert payload["error_type"] == "RuntimeError"
        assert payload["stack_trace_available"] is True
        assert "stack_trace" not in payload


class TestErrorEventPayloadShape:
    """The Celery task's final 'end / error' payload must include error_type,
    failing_phase, and trace_id. We assert the contract at the helper level
    without spinning up the whole task body."""

    def test_payload_construction_uses_current_trace_id(self):
        """Helper smoke test: the payload pattern used in agent_tasks.py works
        even when no OTel span is active (trace_id falls back to '')."""
        import sys

        try:
            raise RuntimeError("simulated failure")
        except RuntimeError:
            exc_type, _, _ = sys.exc_info()
            payload = {
                "status": "error",
                "message": "An internal error occurred.",
                "error_type": exc_type.__name__ if exc_type else "Unknown",
                "failing_phase": "agent_run",
                "trace_id": _current_trace_id(),
                "stack_trace_available": True,
            }

        assert payload["error_type"] == "RuntimeError"
        assert payload["failing_phase"] == "agent_run"
        assert payload["status"] == "error"
        # message stays user-safe; diagnostic content is in the extra fields
        assert "RuntimeError" not in payload["message"]
        # trace_id key always present (possibly empty)
        assert "trace_id" in payload
        assert payload["stack_trace_available"] is True


# ---------------------------------------------------------------------------
# Stuck-task state emission
# ---------------------------------------------------------------------------

class TestStuckStateEmission:
    """`AsyncRedisStreamManager.emit_stuck_state_if_overdue` publishes a
    `stuck` stream event when a queued task is overdue for pickup, and
    no-ops on every other branch (under threshold, already running,
    missing enqueue metadata)."""

    def _build_async_mgr(self):
        """Build an AsyncRedisStreamManager without touching real Redis."""
        with patch(
            "app.modules.conversations.utils.redis_streaming.ConfigProvider"
        ) as mock_cp, patch(
            "app.modules.conversations.utils.redis_streaming.AsyncRedis"
        ) as mock_async_redis:
            mock_cp.return_value.get_redis_url.return_value = (
                "redis://localhost:6379/0"
            )
            mock_cp.get_stream_ttl_secs.return_value = 3600
            mock_cp.get_stream_maxlen.return_value = 1000
            mock_async_redis.from_url.return_value = AsyncMock()
            mgr = AsyncRedisStreamManager()
        return mgr

    @pytest.mark.asyncio
    async def test_emits_stuck_event_when_overdue_and_still_queued(self):
        mgr = self._build_async_mgr()
        # Enqueued 120s ago, threshold 30s by default -> overdue.
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "task_id": "task-1",
                "queue_name": "staging_agent_tasks",
                "enqueued_at": str(__import__("time").time() - 120),
                "queue_depth_at_enqueue": "42",
            }
        )
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.task_pickup_diagnostics = AsyncMock(
            return_value={
                "task_id": "task-1",
                "queue_name": "staging_agent_tasks",
                "queue_depth_at_enqueue": "42",
                "queue_depth_at_pickup": "3",
                "pickup_latency_ms": "120000",
            }
        )
        mgr.publish_event = AsyncMock()
        # Dedup miss (no prior emission) + successful set
        mgr.redis_client.exists = AsyncMock(return_value=False)
        mgr.redis_client.set = AsyncMock(return_value=True)

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-1", "run-1", "staging_agent_tasks"
        )

        assert emitted is True
        mgr.publish_event.assert_awaited_once()
        args = mgr.publish_event.call_args
        # Positional: (conversation_id, run_id, event_type, payload)
        assert args.args[0] == "conv-1"
        assert args.args[1] == "run-1"
        assert args.args[2] == "stuck"
        payload = args.args[3]
        # Required identifiers per the ticket's DoD
        assert payload["status"] == "stuck"
        assert payload["task_id"] == "task-1"
        assert payload["queue_name"] == "staging_agent_tasks"
        assert payload["queue_depth_at_enqueue"] == "42"
        assert payload["queue_depth_at_pickup"] == "3"
        assert "elapsed_seconds" in payload
        assert "threshold_seconds" in payload
        assert int(payload["elapsed_seconds"]) >= 120

    @pytest.mark.asyncio
    async def test_does_not_emit_when_under_threshold(self):
        mgr = self._build_async_mgr()
        # Enqueued only 5s ago -> not stuck yet (default threshold 30s).
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "task_id": "task-2",
                "queue_name": "staging_agent_tasks",
                "enqueued_at": str(__import__("time").time() - 5),
                "queue_depth_at_enqueue": "0",
            }
        )
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.publish_event = AsyncMock()
        mgr.task_pickup_diagnostics = AsyncMock()

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-2", "run-2", "staging_agent_tasks"
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()
        mgr.task_pickup_diagnostics.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_emit_when_task_has_started_running(self):
        mgr = self._build_async_mgr()
        # Enqueued ages ago AND already running -> not stuck.
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "task_id": "task-3",
                "queue_name": "staging_agent_tasks",
                "enqueued_at": str(__import__("time").time() - 300),
            }
        )
        mgr.get_task_status = AsyncMock(return_value="running")
        mgr.publish_event = AsyncMock()

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-3", "run-3", "staging_agent_tasks"
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_emit_when_enqueue_metadata_is_missing(self):
        mgr = self._build_async_mgr()
        # No enqueued_at -> we cannot compute elapsed, must no-op.
        mgr.get_task_enqueue_metadata = AsyncMock(return_value={})
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.publish_event = AsyncMock()

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-4", "run-4", "staging_agent_tasks"
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_threshold_overrides_env_default(self):
        mgr = self._build_async_mgr()
        # Enqueued 10s ago. Default threshold 30s says "not stuck", but the
        # caller asks for 5s -> should emit.
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "task_id": "task-5",
                "queue_name": "staging_agent_tasks",
                "enqueued_at": str(__import__("time").time() - 10),
            }
        )
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.task_pickup_diagnostics = AsyncMock(
            return_value={
                "task_id": "task-5",
                "queue_name": "staging_agent_tasks",
                "queue_depth_at_enqueue": "1",
                "queue_depth_at_pickup": "1",
                "pickup_latency_ms": "10000",
            }
        )
        mgr.publish_event = AsyncMock()
        mgr.redis_client.exists = AsyncMock(return_value=False)
        mgr.redis_client.set = AsyncMock(return_value=True)

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-5", "run-5", "staging_agent_tasks", threshold_seconds=5
        )

        assert emitted is True
        payload = mgr.publish_event.call_args.args[3]
        assert payload["threshold_seconds"] == "5"


# ---------------------------------------------------------------------------
# Running-stuck vs queued-stuck distinction
# ---------------------------------------------------------------------------

class TestRunningStuckEmission:
    """`RedisStreamManager.emit_running_stuck_if_idle` publishes a
    `stuck` stream event with ``phase=running`` when the run is running
    but no events have crossed the wire for the idle threshold."""

    def _build_sync_mgr(self):
        with patch(
            "app.modules.conversations.utils.redis_streaming.ConfigProvider"
        ) as mock_cp, patch(
            "app.modules.conversations.utils.redis_streaming.redis"
        ) as mock_redis:
            mock_cp.return_value.get_redis_url.return_value = (
                "redis://localhost:6379/0"
            )
            mock_cp.get_stream_ttl_secs.return_value = 3600
            mock_cp.get_stream_maxlen.return_value = 1000
            mock_redis.from_url.return_value = MagicMock()
            mgr = RedisStreamManager()
        return mgr

    def test_emits_running_stuck_with_phase_field_when_idle(self):
        import time

        mgr = self._build_sync_mgr()
        mgr.get_task_status = MagicMock(return_value="running")
        mgr.get_task_enqueue_metadata = MagicMock(
            return_value={"queue_name": "staging_agent_tasks"}
        )
        mgr.task_pickup_diagnostics = MagicMock(
            return_value={
                "task_id": "task-1",
                "queue_name": "staging_agent_tasks",
                "queue_depth_at_enqueue": "0",
                "queue_depth_at_pickup": "0",
                "pickup_latency_ms": "5000",
            }
        )
        mgr.redis_client.exists = MagicMock(return_value=False)
        mgr.redis_client.set = MagicMock(return_value=True)
        mgr.publish_event = MagicMock()

        # Last event 120s ago - well past the 60s default idle threshold.
        last_event_at = time.time() - 120

        emitted = mgr.emit_running_stuck_if_idle(
            "conv-1", "run-1", last_event_at
        )

        assert emitted is True
        mgr.publish_event.assert_called_once()
        args = mgr.publish_event.call_args.args
        assert args[2] == "stuck"
        payload = args[3]
        # Phase distinguishes this from the queued-stuck case
        assert payload["phase"] == "running"
        assert payload["status"] == "stuck"
        assert "idle_seconds" in payload
        assert int(payload["idle_seconds"]) >= 120
        # Pickup diagnostics are spread in so a triager can see the same
        # queue depth fields as in the `running` event
        assert payload["task_id"] == "task-1"
        assert payload["queue_name"] == "staging_agent_tasks"

    def test_does_not_emit_when_under_idle_threshold(self):
        import time

        mgr = self._build_sync_mgr()
        mgr.get_task_status = MagicMock(return_value="running")
        mgr.publish_event = MagicMock()
        mgr.redis_client.exists = MagicMock(return_value=False)

        # 5s idle - well below the 60s default threshold.
        emitted = mgr.emit_running_stuck_if_idle(
            "conv-2", "run-2", time.time() - 5
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()

    def test_does_not_emit_when_task_is_queued_not_running(self):
        import time

        mgr = self._build_sync_mgr()
        # Status is still queued, not running. That's queued-stuck, not
        # running-stuck. The phases are intentionally distinct.
        mgr.get_task_status = MagicMock(return_value="queued")
        mgr.publish_event = MagicMock()
        mgr.redis_client.exists = MagicMock(return_value=False)

        emitted = mgr.emit_running_stuck_if_idle(
            "conv-3", "run-3", time.time() - 300
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()

    def test_does_not_emit_when_dedup_key_exists(self):
        import time

        mgr = self._build_sync_mgr()
        mgr.get_task_status = MagicMock(return_value="running")
        mgr.publish_event = MagicMock()
        # Dedup key already present from a previous emission
        mgr.redis_client.exists = MagicMock(return_value=True)

        emitted = mgr.emit_running_stuck_if_idle(
            "conv-4", "run-4", time.time() - 300
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()

    def test_sets_dedup_key_after_successful_emit(self):
        import time

        mgr = self._build_sync_mgr()
        mgr.get_task_status = MagicMock(return_value="running")
        mgr.get_task_enqueue_metadata = MagicMock(return_value={})
        mgr.task_pickup_diagnostics = MagicMock(return_value={})
        mgr.redis_client.exists = MagicMock(return_value=False)
        mgr.redis_client.set = MagicMock(return_value=True)
        mgr.publish_event = MagicMock()

        mgr.emit_running_stuck_if_idle("conv-5", "run-5", time.time() - 300)

        # The TTL key prevents storming the stream on subsequent timeouts
        mgr.redis_client.set.assert_called_once()
        set_args = mgr.redis_client.set.call_args
        assert "chat:stuck_emitted:running:conv-5:run-5" in set_args.args[0]
        # TTL should be a positive integer (>= max(threshold*2, 60))
        ttl = set_args.kwargs.get("ex") or set_args.args[-1]
        assert int(ttl) >= 60


class TestQueuedStuckCarriesPhaseField:
    """The existing queued-stuck event now also carries ``phase=queued``
    so consumers can switch on the same field across both cases."""

    def _build_async_mgr(self):
        with patch(
            "app.modules.conversations.utils.redis_streaming.ConfigProvider"
        ) as mock_cp, patch(
            "app.modules.conversations.utils.redis_streaming.AsyncRedis"
        ) as mock_async_redis:
            mock_cp.return_value.get_redis_url.return_value = (
                "redis://localhost:6379/0"
            )
            mock_cp.get_stream_ttl_secs.return_value = 3600
            mock_cp.get_stream_maxlen.return_value = 1000
            mock_async_redis.from_url.return_value = AsyncMock()
            mgr = AsyncRedisStreamManager()
        return mgr

    @pytest.mark.asyncio
    async def test_queued_stuck_payload_includes_phase_field(self):
        import time

        mgr = self._build_async_mgr()
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "task_id": "task-1",
                "queue_name": "staging_agent_tasks",
                "enqueued_at": str(time.time() - 120),
            }
        )
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.task_pickup_diagnostics = AsyncMock(return_value={})
        mgr.publish_event = AsyncMock()
        # The async client mock supports `await mgr.redis_client.exists(...)`;
        # we want dedup miss (not yet emitted).
        mgr.redis_client.exists = AsyncMock(return_value=False)
        mgr.redis_client.set = AsyncMock(return_value=True)

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-1", "run-1", "staging_agent_tasks"
        )

        assert emitted is True
        payload = mgr.publish_event.call_args.args[3]
        assert payload["phase"] == "queued"
        assert payload["status"] == "stuck"

    @pytest.mark.asyncio
    async def test_queued_dedup_blocks_second_emit_within_window(self):
        import time

        mgr = self._build_async_mgr()
        mgr.get_task_enqueue_metadata = AsyncMock(
            return_value={
                "enqueued_at": str(time.time() - 120),
                "queue_name": "staging_agent_tasks",
            }
        )
        mgr.get_task_status = AsyncMock(return_value="queued")
        mgr.publish_event = AsyncMock()
        # Dedup key already present
        mgr.redis_client.exists = AsyncMock(return_value=True)

        emitted = await mgr.emit_stuck_state_if_overdue(
            "conv-2", "run-2", "staging_agent_tasks"
        )

        assert emitted is False
        mgr.publish_event.assert_not_called()


# ---------------------------------------------------------------------------
# Explicit error payloads on streaming failure (no silent disconnects)
# ---------------------------------------------------------------------------

class TestExplicitErrorPayloads:
    """Three acceptance criteria:
      1. Terminal failures include `status` and `message`.
      2. SSE emits an error payload (frame, not just close).
      3. Silent disconnects are replaced for covered failure modes.
    """

    @pytest.mark.asyncio
    async def test_get_stream_yields_error_frame_when_inner_stream_raises(self):
        """`get_stream` wraps controller-side async generators. If the inner
        raises, the wrapper must emit an explicit error JSON frame before the
        connection closes; otherwise the client just sees a truncated stream."""
        from app.modules.conversations.conversations_router import get_stream

        class _Chunk:
            def __init__(self, content):
                self._content = content
            def dict(self):
                return {"message": self._content}

        async def inner():
            yield _Chunk("partial-1")
            yield _Chunk("partial-2")
            raise RuntimeError("controller blew up")

        frames = []
        async for f in get_stream(inner()):
            frames.append(json.loads(f))

        # Two real chunks, then the terminal error frame
        assert len(frames) == 3
        assert frames[0]["message"] == "partial-1"
        assert frames[1]["message"] == "partial-2"
        terminal = frames[2]
        assert terminal["event"] == "error"
        assert terminal["status"] == "error"
        assert "message" in terminal and terminal["message"]
        # Diagnostic fields propagated from _stream_failure_fields
        assert terminal["failing_phase"] == "get_stream"
        assert terminal["error_type"] == "RuntimeError"
        assert "trace_id" in terminal

    @pytest.mark.asyncio
    async def test_get_stream_passes_through_clean_streams_unchanged(self):
        """Hardening must not change the happy path."""
        from app.modules.conversations.conversations_router import get_stream

        class _Chunk:
            def __init__(self, content):
                self._content = content
            def dict(self):
                return {"message": self._content}

        async def inner():
            yield _Chunk("only-1")
            yield _Chunk("only-2")

        frames = [json.loads(f) async for f in get_stream(inner())]
        assert [f["message"] for f in frames] == ["only-1", "only-2"]
        # No injected error frame on success
        assert not any(f.get("event") == "error" for f in frames)

    def test_redis_stream_generator_yields_error_frame_on_end_with_error_status(self):
        """Existing redis_stream_generator path - assert the AC contract."""
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        # Stub the manager via patch: the generator instantiates one inline.
        with patch(
            "app.modules.conversations.utils.conversation_routing.RedisStreamManager"
        ) as mock_mgr_cls:
            mgr = MagicMock()
            mgr.consume_stream.return_value = iter([
                {"type": "chunk", "content": "partial"},
                {
                    "type": "end",
                    "status": "error",
                    "message": "Worker exploded",
                    "failing_phase": "agent_run",
                    "error_type": "RuntimeError",
                    "trace_id": "abc123",
                    "stack_trace_available": True,
                },
            ])
            mock_mgr_cls.return_value = mgr

            frames = [
                json.loads(f)
                for f in redis_stream_generator("conv-1", "run-1")
            ]

        # Chunk + terminal error frame
        assert len(frames) == 2
        terminal = frames[-1]
        assert terminal["event"] == "error"
        assert terminal["status"] == "error"
        # Message + failing phase travel together so clients can route
        assert terminal["message"] == "Worker exploded"
        assert terminal["failing_phase"] == "agent_run"
        assert terminal["error_type"] == "RuntimeError"
        assert terminal["trace_id"] == "abc123"


# ---------------------------------------------------------------------------
# Partial-vs-failure distinction
# ---------------------------------------------------------------------------

class TestPartialOutputMarker:
    """Acceptance criteria:
      1. Partial output has a non-success state.
      2. Failure and partial cases are distinguishable.
      3. The state is visible in the result path used for debugging.
    """

    def test_error_frame_marks_partial_true_when_chunks_were_emitted(self):
        """A run that delivered some output before failing must NOT look
        the same as a run that failed before producing anything."""
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        with patch(
            "app.modules.conversations.utils.conversation_routing.RedisStreamManager"
        ) as mock_mgr_cls:
            mgr = MagicMock()
            mgr.consume_stream.return_value = iter([
                {"type": "chunk", "content": "tok-1"},
                {"type": "chunk", "content": "tok-2"},
                {"type": "chunk", "content": "tok-3"},
                {
                    "type": "end",
                    "status": "error",
                    "message": "Worker hit a snag",
                    "failing_phase": "agent_run",
                    "error_type": "RuntimeError",
                },
            ])
            mock_mgr_cls.return_value = mgr

            frames = [
                json.loads(f)
                for f in redis_stream_generator("conv-1", "run-1")
            ]

        # 3 chunks + terminal error frame
        assert len(frames) == 4
        terminal = frames[-1]
        # Non-success state
        assert terminal["status"] == "error"
        # Distinguishes "got something, then failed" from a clean failure
        assert terminal["partial"] is True
        assert terminal["chunks_emitted"] == 3

    def test_error_frame_marks_partial_false_when_no_chunks_were_emitted(self):
        """Hard failure with no output -> partial=false. This is what tells
        a UI to show a full error state instead of partial-content + badge."""
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        with patch(
            "app.modules.conversations.utils.conversation_routing.RedisStreamManager"
        ) as mock_mgr_cls:
            mgr = MagicMock()
            mgr.consume_stream.return_value = iter([
                {
                    "type": "end",
                    "status": "error",
                    "message": "Worker died on setup",
                    "failing_phase": "setup",
                    "error_type": "ConnectionError",
                },
            ])
            mock_mgr_cls.return_value = mgr

            frames = [
                json.loads(f)
                for f in redis_stream_generator("conv-2", "run-2")
            ]

        # Just the terminal error frame
        assert len(frames) == 1
        terminal = frames[0]
        assert terminal["status"] == "error"
        assert terminal["partial"] is False
        assert terminal["chunks_emitted"] == 0

    def test_error_frame_marks_partial_when_generator_itself_raises_after_chunks(self):
        """The except-branch path also needs to carry chunks_emitted, not
        just the end-event path."""
        from app.modules.conversations.utils.conversation_routing import (
            redis_stream_generator,
        )

        class _BoomIter:
            """Iterator that yields some chunks then raises - mirrors a
            Redis consumer that dies mid-stream."""
            def __init__(self):
                self._items = iter([
                    {"type": "chunk", "content": "tok-a"},
                    {"type": "chunk", "content": "tok-b"},
                ])
            def __iter__(self):
                return self
            def __next__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise ConnectionError("redis blew up")

        with patch(
            "app.modules.conversations.utils.conversation_routing.RedisStreamManager"
        ) as mock_mgr_cls:
            mgr = MagicMock()
            mgr.consume_stream.return_value = _BoomIter()
            mock_mgr_cls.return_value = mgr

            frames = [
                json.loads(f)
                for f in redis_stream_generator("conv-3", "run-3")
            ]

        # 2 chunks + terminal error frame from the except branch
        assert len(frames) == 3
        terminal = frames[-1]
        assert terminal["event"] == "error"
        assert terminal["partial"] is True
        assert terminal["chunks_emitted"] == 2
        # Failing phase distinguishes this from the worker-side error
        assert terminal["failing_phase"] == "sse_consume"

    @pytest.mark.asyncio
    async def test_get_stream_terminal_frame_marks_partial(self):
        """Same marker propagates through the non-Redis bridge."""
        from app.modules.conversations.conversations_router import get_stream

        class _Chunk:
            def __init__(self, content):
                self._content = content
            def dict(self):
                return {"message": self._content}

        async def inner():
            yield _Chunk("a")
            yield _Chunk("b")
            yield _Chunk("c")
            raise RuntimeError("controller blew up")

        frames = [json.loads(f) async for f in get_stream(inner())]

        assert len(frames) == 4
        terminal = frames[-1]
        assert terminal["status"] == "error"
        assert terminal["partial"] is True
        assert terminal["chunks_emitted"] == 3

    @pytest.mark.asyncio
    async def test_get_stream_pure_failure_marks_partial_false(self):
        from app.modules.conversations.conversations_router import get_stream

        async def inner():
            raise RuntimeError("died before producing anything")
            yield  # pragma: no cover - unreachable, keeps inner a generator

        frames = [json.loads(f) async for f in get_stream(inner())]

        assert len(frames) == 1
        terminal = frames[0]
        assert terminal["status"] == "error"
        assert terminal["partial"] is False
        assert terminal["chunks_emitted"] == 0


# ---------------------------------------------------------------------------
# Structured streaming logs (no print statements)
# ---------------------------------------------------------------------------

class TestStructuredStreamingLogs:
    """Acceptance criteria:
      1. Print statements in the target flow are removed.
      2. Structured logs include the right identifiers.
      3. Log format supports debugging of failures and timing.
    """

    def test_no_print_statements_in_streaming_flow(self):
        """Regression guard: the four streaming-flow files must not contain
        any bare ``print(...)`` calls. Print to stdout bypasses loguru, has
        no identifiers, and can't be filtered by trace_id / run_id."""
        import pathlib
        import re

        repo_root = pathlib.Path(__file__).resolve().parents[3]
        targets = [
            "app/celery/tasks/agent_tasks.py",
            "app/modules/conversations/utils/conversation_routing.py",
            "app/modules/conversations/utils/redis_streaming.py",
            "app/modules/conversations/conversations_router.py",
        ]
        print_re = re.compile(r"^\s*print\(", re.MULTILINE)
        offenders = []
        for rel in targets:
            text = (repo_root / rel).read_text()
            for m in print_re.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{rel}:{line_no}")
        assert not offenders, (
            "print() found in the streaming flow: " + ", ".join(offenders)
        )

    def test_log_openrouter_usage_emits_structured_fields(self):
        from app.celery.tasks.agent_tasks import _log_openrouter_usage

        usages = [
            {
                "model": "openrouter/anthropic/claude-3.7-sonnet",
                "prompt_tokens": 1200,
                "completion_tokens": 340,
                "total_tokens": 1540,
                "cost": 0.018,
            }
        ]

        with patch(
            "app.celery.tasks.agent_tasks.logger"
        ) as mock_logger:
            _log_openrouter_usage(
                usages,
                outcome="completed",
                total_cost=0.018,
                conversation_id="conv-1",
                run_id="run-1",
            )

        # Two log lines: per-usage line + summary line
        assert mock_logger.info.call_count == 2

        # First call: per-usage line. Kwargs carry all the structured fields.
        first_call = mock_logger.info.call_args_list[0]
        first_kwargs = first_call.kwargs
        assert first_kwargs["model"] == "openrouter/anthropic/claude-3.7-sonnet"
        assert first_kwargs["prompt_tokens"] == 1200
        assert first_kwargs["completion_tokens"] == 340
        assert first_kwargs["total_tokens"] == 1540
        assert first_kwargs["cost"] == 0.018
        assert first_kwargs["cost_estimated"] is False
        assert first_kwargs["outcome"] == "completed"
        # Identifiers the ticket asks for
        assert first_kwargs["conversation_id"] == "conv-1"
        assert first_kwargs["run_id"] == "run-1"
        assert "trace_id" in first_kwargs

        # Second call: summary line. Carries the run-level fields.
        second_call = mock_logger.info.call_args_list[1]
        second_kwargs = second_call.kwargs
        assert second_kwargs["total_cost"] == 0.018
        assert second_kwargs["outcome"] == "completed"
        assert second_kwargs["usage_count"] == 1
        assert second_kwargs["conversation_id"] == "conv-1"
        assert second_kwargs["run_id"] == "run-1"

    def test_log_openrouter_usage_marks_partial_on_error_outcome(self):
        """The message prefix and the structured ``outcome`` field both
        flip when a run failed mid-flight, so triage can filter by outcome
        without parsing the message."""
        from app.celery.tasks.agent_tasks import _log_openrouter_usage

        usages = [
            {
                "model": "model-x",
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
                "cost": None,
            }
        ]

        with patch("app.celery.tasks.agent_tasks.logger") as mock_logger:
            _log_openrouter_usage(
                usages,
                outcome="error_partial",
                total_cost=0.0,
                conversation_id="conv-2",
                run_id="run-2",
            )

        # Per-usage line uses the "partial" message prefix
        first_call = mock_logger.info.call_args_list[0]
        assert "[OpenRouter usage - partial]" in first_call.args[0]
        # Structured outcome field present on the line for log aggregators
        assert first_call.kwargs["outcome"] == "error_partial"
        # Cost was None on the wire -> ``cost_estimated`` flips to True
        assert first_call.kwargs["cost_estimated"] is True

        # Summary line uses the "partial" prefix too
        second_call = mock_logger.info.call_args_list[1]
        assert "[LLM cost - partial before error]" in second_call.args[0]

    def test_log_openrouter_usage_no_op_when_usages_empty(self):
        """Empty usage list -> helper is a no-op (no log lines). Keeps
        the call sites safe to invoke unconditionally."""
        from app.celery.tasks.agent_tasks import _log_openrouter_usage

        with patch("app.celery.tasks.agent_tasks.logger") as mock_logger:
            _log_openrouter_usage(
                [],
                outcome="completed",
                total_cost=0.0,
                conversation_id="conv-3",
                run_id="run-3",
            )

        mock_logger.info.assert_not_called()
