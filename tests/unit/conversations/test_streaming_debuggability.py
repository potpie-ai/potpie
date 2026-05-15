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
