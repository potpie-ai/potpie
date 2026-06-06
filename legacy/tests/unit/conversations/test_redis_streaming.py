"""Unit tests for RedisStreamManager (sync) and AsyncRedisStreamManager."""
from unittest.mock import MagicMock, patch

import pytest

from app.modules.conversations.utils.redis_streaming import (
    RedisStreamManager,
    AsyncRedisStreamManager,
)


pytestmark = pytest.mark.unit


class TestRedisStreamManagerStreamKey:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_stream_key_format(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        mock_redis.from_url.return_value = MagicMock()
        mgr = RedisStreamManager()
        assert mgr.stream_key("conv-1", "run-2") == "chat:stream:conv-1:run-2"


class TestRedisStreamManagerPublishEvent:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_publish_event_calls_xadd_and_expire(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        mgr.publish_event("c1", "r1", "chunk", {"text": "hello"})
        assert client.xadd.called
        assert client.expire.called

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_publish_event_serializes_dict_payload(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        mgr.publish_event("c1", "r1", "chunk", {"nested": {"a": 1}})
        call_args = client.xadd.call_args
        assert call_args is not None
        event_data = call_args[0][1]
        assert event_data["type"] == "chunk"
        assert event_data["conversation_id"] == "c1"
        assert event_data["run_id"] == "r1"


class TestRedisStreamManagerFormatEvent:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_format_event_decodes_bytes(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        mock_redis.from_url.return_value = MagicMock()
        mgr = RedisStreamManager()
        event_id = b"1234-0"
        event_data = {b"type": b"chunk", b"text": b"hello"}
        out = mgr._format_event(event_id, event_data)
        assert out["stream_id"] == "1234-0"
        assert out["type"] == "chunk"
        assert out["text"] == "hello"

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_format_event_parses_json_suffix_keys(self, mock_redis, mock_cp):
        import json
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        mock_redis.from_url.return_value = MagicMock()
        mgr = RedisStreamManager()
        event_data = {b"type": b"chunk", b"tool_calls_json": b'[{"name":"x"}]'}
        out = mgr._format_event("0-0", event_data)
        assert out["tool_calls"] == [{"name": "x"}]


class TestRedisStreamManagerConsumeStream:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_fresh_connect_replays_existing_stream_from_start(self, mock_redis, mock_cp):
        """Hatchet can publish chunks before HTTP attaches; do not skip to latest."""
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.exists.return_value = True
        client.xrange.return_value = [
            (
                b"1-0",
                {
                    b"type": b"chunk",
                    b"content": b"early",
                    b"tool_calls_json": b'[{"tool_name":"search"}]',
                },
            ),
            (b"2-0", {b"type": b"end", b"status": b"completed"}),
        ]
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        events = list(mgr.consume_stream("c1", "r1", cursor=None))
        types = [e.get("type") for e in events]
        assert types == ["chunk", "end"]
        assert events[0].get("content") == "early"
        assert events[0].get("tool_calls") == [{"tool_name": "search"}]
        client.xrevrange.assert_not_called()


class TestRedisStreamManagerCancellation:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_check_cancellation_true(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.get.return_value = b"true"
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        assert mgr.check_cancellation("c1", "r1") is True
        client.get.assert_called_once_with("cancel:c1:r1")

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_check_cancellation_false(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.get.return_value = None
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        assert mgr.check_cancellation("c1", "r1") is False

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_set_cancellation(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        mgr.set_cancellation("c1", "r1")
        client.set.assert_called_once_with("cancel:c1:r1", "true", ex=300)


class TestRedisStreamManagerTaskStatus:
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_set_and_get_task_status(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.get.return_value = b"running"
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        mgr.set_task_status("c1", "r1", "running")
        client.set.assert_called_with("task:status:c1:r1", "running", ex=600)
        assert mgr.get_task_status("c1", "r1") == "running"

    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.redis")
    def test_get_task_status_none(self, mock_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        client = MagicMock()
        client.get.return_value = None
        mock_redis.from_url.return_value = client
        mgr = RedisStreamManager()
        assert mgr.get_task_status("c1", "r1") is None


class TestAsyncRedisStreamManagerStreamKey:
    @pytest.mark.asyncio
    @patch("app.modules.conversations.utils.redis_streaming.ConfigProvider")
    @patch("app.modules.conversations.utils.redis_streaming.AsyncRedis")
    async def test_async_stream_key_format(self, mock_async_redis, mock_cp):
        mock_cp.return_value.get_redis_url.return_value = "redis://localhost:6379/0"
        mock_cp.get_stream_ttl_secs.return_value = 3600
        mock_cp.get_stream_maxlen.return_value = 1000
        mock_async_redis.from_url.return_value = MagicMock()
        mgr = AsyncRedisStreamManager()
        assert mgr.stream_key("conv-a", "run-b") == "chat:stream:conv-a:run-b"
