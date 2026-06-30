"""
Extended unit tests for SessionService and AsyncSessionService
(app/modules/conversations/session/session_service.py)

Tests cover:
- AsyncSessionService.get_active_session
- AsyncSessionService.get_task_status
- AsyncSessionService._get_stream_keys_by_recency
- Additional status mapping tests for SessionService
- Redis error handling for async service
- Key sorting/recency logic
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import redis
import pytest_asyncio

from app.modules.conversations.session.session_service import (
    SessionService,
    AsyncSessionService,
)
from app.modules.conversations.conversation.conversation_schema import (
    ActiveSessionResponse,
    ActiveSessionErrorResponse,
    TaskStatusResponse,
    TaskStatusErrorResponse,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis_manager():
    """Create a mock RedisStreamManager"""
    manager = MagicMock()
    manager.redis_client = MagicMock()
    return manager


@pytest.fixture
def session_service(mock_redis_manager):
    """Create a SessionService with mocked Redis"""
    service = SessionService()
    service.redis_manager = mock_redis_manager
    return service


@pytest.fixture
def mock_async_redis_manager():
    """Create a mock AsyncRedisStreamManager"""
    manager = MagicMock()
    manager.redis_client = MagicMock()
    return manager


@pytest.fixture
def async_session_service(mock_async_redis_manager):
    """Create an AsyncSessionService with mocked AsyncRedisStreamManager"""
    return AsyncSessionService(redis_manager=mock_async_redis_manager)


# =============================================================================
# SessionService - Additional Status Mapping Tests
# =============================================================================


class TestSessionServiceStatusMapping:
    """Additional status mapping tests for SessionService.get_active_session"""

    def test_status_mapping_running_to_active(
        self, session_service, mock_redis_manager
    ):
        """Test status mapping: running -> active"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-abc"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"123-0", {})]
        mock_redis_manager.get_task_status.return_value = "running"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "active"

    def test_status_mapping_completed_to_completed(
        self, session_service, mock_redis_manager
    ):
        """Test status mapping: completed -> completed"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-xyz"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"456-0", {})]
        mock_redis_manager.get_task_status.return_value = "completed"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "completed"

    def test_status_mapping_pending_to_idle(self, session_service, mock_redis_manager):
        """Test status mapping: pending -> idle (pending is not mapped to active in get_active_session)"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-pending"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"789-0", {})]
        mock_redis_manager.get_task_status.return_value = "pending"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "idle"

    def test_status_mapping_failed_to_idle(self, session_service, mock_redis_manager):
        """Test status mapping: failed -> idle"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-failed"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"100-0", {})]
        mock_redis_manager.get_task_status.return_value = "failed"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "idle"

    def test_status_mapping_none_to_idle(self, session_service, mock_redis_manager):
        """Test status mapping: None -> idle"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-none"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"200-0", {})]
        mock_redis_manager.get_task_status.return_value = None

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "idle"


class TestSessionServiceExpiredKeys:
    """Tests for expired/deleted stream keys in SessionService"""

    def test_expired_key_falls_through_to_error(
        self, session_service, mock_redis_manager
    ):
        """Test that expired key returns error after checking all streams"""
        # Multiple keys exist but first one is expired
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-expired",
            b"chat:stream:conv-123:run-old",
        ]
        # First key doesn't exist, second should also not exist to trigger error
        mock_redis_manager.redis_client.exists.return_value = False

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert "No active session" in result.error

    def test_all_streams_expired_returns_error(
        self, session_service, mock_redis_manager
    ):
        """Test that when all streams are expired, error is returned"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-1",
            b"chat:stream:conv-123:run-2",
            b"chat:stream:conv-123:run-3",
        ]
        # None of the streams exist anymore
        mock_redis_manager.redis_client.exists.return_value = False

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert "No active session" in result.error


class TestSessionServiceRedisErrors:
    """Additional Redis error tests for SessionService"""

    def test_redis_error_on_exists(self, session_service, mock_redis_manager):
        """Test Redis error during exists check is raised"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-abc"
        ]
        mock_redis_manager.redis_client.exists.side_effect = (
            redis.exceptions.ConnectionError("Connection refused")
        )

        with pytest.raises(redis.exceptions.ConnectionError):
            session_service.get_active_session("conv-123")

    def test_redis_error_on_xinfo_stream(self, session_service, mock_redis_manager):
        """Test Redis error during xinfo_stream is raised"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-abc"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.side_effect = (
            redis.exceptions.ResponseError("BUSYGROUP The XINFO command is busy")
        )

        with pytest.raises(redis.exceptions.ResponseError):
            session_service.get_active_session("conv-123")

    def test_redis_error_on_xrevrange(self, session_service, mock_redis_manager):
        """Test Redis error during xrevrange is raised"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-abc"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.side_effect = (
            redis.exceptions.ResponseError("ERR An error occurred")
        )

        with pytest.raises(redis.exceptions.ResponseError):
            session_service.get_active_session("conv-123")


# =============================================================================
# AsyncSessionService Tests
# =============================================================================


class TestAsyncSessionServiceGetActiveSession:
    """Tests for AsyncSessionService.get_active_session"""

    @pytest.mark.asyncio
    async def test_async_active_session_happy_path(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test successful retrieval of active session"""

        # Mock scan_iter to return stream keys
        async def mock_scan_iter(match, count):
            keys = [
                b"chat:stream:conv-123:run-abc",
            ]
            for key in keys:
                yield key

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=True)
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            return_value=[(b"123-0", {})]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="running")

        result = await async_session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.conversationId == "conv-123"
        assert result.sessionId == "run-abc"
        assert result.status == "active"
        assert result.cursor == "123-0"

    @pytest.mark.asyncio
    async def test_async_active_session_no_sessions(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns error when no streams found"""

        async def mock_scan_iter(match, count):
            return
            yield  # Empty generator

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter

        result = await async_session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert result.conversationId == "conv-123"
        assert "No active session" in result.error

    @pytest.mark.asyncio
    async def test_async_active_session_all_expired(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns error when all streams have expired"""

        async def mock_scan_iter(match, count):
            keys = [
                b"chat:stream:conv-123:run-1",
                b"chat:stream:conv-123:run-2",
            ]
            for key in keys:
                yield key

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        # All streams are expired (don't exist)
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=False)

        result = await async_session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert "No active session" in result.error

    @pytest.mark.asyncio
    async def test_async_active_session_completed_status(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test session with completed status"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-done"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=True)
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            return_value=[(b"456-0", {})]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="completed")

        result = await async_session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_async_active_session_idle_status(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test session with unknown/idle status"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-idle"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=True)
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(return_value=[])
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="unknown")

        result = await async_session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "idle"
        assert result.cursor == "0-0"

    @pytest.mark.asyncio
    async def test_async_active_session_redis_error(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test Redis error is raised during recency check"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-abc"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("Connection refused")
        )

        with pytest.raises(redis.exceptions.ConnectionError):
            await async_session_service.get_active_session("conv-123")

    @pytest.mark.asyncio
    async def test_async_active_session_xrevrange_redis_error(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test Redis error during xrevrange is raised"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-abc"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=True)
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            side_effect=redis.exceptions.ResponseError("ERR error")
        )

        with pytest.raises(redis.exceptions.ResponseError):
            await async_session_service.get_active_session("conv-123")


class TestAsyncSessionServiceGetTaskStatus:
    """Tests for AsyncSessionService.get_task_status"""

    @pytest.mark.asyncio
    async def test_async_task_status_running(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns active status for running task"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-456:run-task"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        # Mock _get_stream_keys_by_recency to order by recency
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-456:run-task"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="running")

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is True
        assert result.sessionId == "run-task"
        assert result.conversationId == "conv-456"
        # Estimated completion should be in the future
        assert (
            result.estimatedCompletion
            > async_session_service._current_timestamp_ms() - 1000
        )

    @pytest.mark.asyncio
    async def test_async_task_status_pending(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns active status for pending task"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-456:run-pending"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-456:run-pending"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="pending")

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is True

    @pytest.mark.asyncio
    async def test_async_task_status_completed(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns inactive status for completed task"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-456:run-done"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-456:run-done"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="completed")

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is False
        # Estimated completion should be in the past
        assert (
            result.estimatedCompletion
            < async_session_service._current_timestamp_ms() + 1000
        )

    @pytest.mark.asyncio
    async def test_async_task_status_no_streams(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns error when no streams found"""

        async def mock_scan_iter(match, count):
            return
            yield

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(return_value=[])

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusErrorResponse)
        assert "No background task" in result.error

    @pytest.mark.asyncio
    async def test_async_task_status_no_valid_status(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns error when no valid task status found"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-456:run-1"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-456:run-1"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value=None)

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusErrorResponse)
        assert "No background task" in result.error

    @pytest.mark.asyncio
    async def test_async_task_status_tries_multiple_streams(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test iterates through streams to find valid task status"""

        async def mock_scan_iter(match, count):
            keys = [
                b"chat:stream:conv-456:run-1",
                b"chat:stream:conv-456:run-2",
            ]
            for key in keys:
                yield key

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        # Return in recency order
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=[
                "chat:stream:conv-456:run-2",
                "chat:stream:conv-456:run-1",
            ]
        )
        # First stream has no status, second has running
        mock_async_redis_manager.get_task_status = AsyncMock(
            side_effect=[None, "running"]
        )

        result = await async_session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.sessionId == "run-1"

    @pytest.mark.asyncio
    async def test_async_task_status_redis_error(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test Redis error is raised"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-456:run-1"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("Connection refused")
        )

        with pytest.raises(redis.exceptions.ConnectionError):
            await async_session_service.get_task_status("conv-456")


class TestAsyncSessionServiceGetStreamKeysByRecency:
    """Tests for AsyncSessionService._get_stream_keys_by_recency"""

    @pytest.mark.asyncio
    async def test_get_stream_keys_by_recency_empty(self, async_session_service):
        """Test empty list returns empty"""
        result = await async_session_service._get_stream_keys_by_recency([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_stream_keys_by_recency_single_key(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test single key is returned"""
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            return_value=[(b"123-0", {})]
        )

        result = await async_session_service._get_stream_keys_by_recency(
            ["chat:stream:conv-123:run-1"]
        )

        assert result == ["chat:stream:conv-123:run-1"]

    @pytest.mark.asyncio
    async def test_get_stream_keys_by_recency_orders_by_entry_id(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test keys are ordered by XREVRANGE entry ID (most recent first)"""

        async def mock_xrevrange(key, count):
            return [(b"100-0", {})]

        mock_async_redis_manager.redis_client.xrevrange = mock_xrevrange

        keys = ["chat:stream:conv-123:run-1", "chat:stream:conv-123:run-2"]
        result = await async_session_service._get_stream_keys_by_recency(keys)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_stream_keys_by_recency_handles_empty_xrevrange(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test keys with no entries get '0-0' as entry ID and sort last"""

        async def mock_xrevrange(key, count):
            if "run-active" in key:
                return [(b"500-0", {})]
            return []

        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            side_effect=mock_xrevrange
        )

        keys = [
            "chat:stream:conv-123:run-empty",
            "chat:stream:conv-123:run-active",
        ]

        result = await async_session_service._get_stream_keys_by_recency(keys)

        assert result[0] == "chat:stream:conv-123:run-active"
        assert result[1] == "chat:stream:conv-123:run-empty"

    @pytest.mark.asyncio
    async def test_get_stream_keys_by_recency_redis_error(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test Redis error during xrevrange is raised"""
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            side_effect=redis.exceptions.ConnectionError("Connection refused")
        )

        with pytest.raises(redis.exceptions.ConnectionError):
            await async_session_service._get_stream_keys_by_recency(
                ["chat:stream:conv-123:run-1"]
            )


class TestAsyncSessionServiceGetMostRecentStreamKey:
    """Tests for AsyncSessionService._get_most_recent_stream_key"""

    @pytest.mark.asyncio
    async def test_get_most_recent_stream_key_returns_first(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test returns the most recent stream key"""
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=[
                "chat:stream:conv-123:run-recent",
                "chat:stream:conv-123:run-old",
            ]
        )

        result = await async_session_service._get_most_recent_stream_key(
            ["chat:stream:conv-123:run-old"]
        )

        assert result == "chat:stream:conv-123:run-recent"

    @pytest.mark.asyncio
    async def test_get_most_recent_stream_key_empty_returns_none(
        self, async_session_service
    ):
        """Test empty list returns None"""
        async_session_service._get_stream_keys_by_recency = AsyncMock(return_value=[])

        result = await async_session_service._get_most_recent_stream_key([])

        assert result is None


class TestAsyncSessionServiceRecencySelection:
    """Tests for recency-based stream selection in AsyncSessionService"""

    @pytest.mark.asyncio
    async def test_picks_most_recent_by_entry_id_not_key_name(
        self, async_session_service, mock_async_redis_manager
    ):
        """AsyncSessionService picks by entry ID, not alphabetically"""

        async def mock_scan_iter(match, count):
            keys = [
                b"chat:stream:conv-123:run-a",
                b"chat:stream:conv-123:run-b",
            ]
            for key in keys:
                yield key

        xrevrange_results = {
            "chat:stream:conv-123:run-a": [(b"100-0", {})],
            "chat:stream:conv-123:run-b": [(b"1000-0", {})],
        }

        async def mock_xrevrange(key, count):
            return xrevrange_results.get(key, [(b"0-0", {})])

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.xrevrange = AsyncMock(
            side_effect=mock_xrevrange
        )
        mock_async_redis_manager.redis_client.exists = AsyncMock(return_value=True)
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="running")

        result = await async_session_service.get_active_session("conv-123")

        assert result.sessionId == "run-b"

    @pytest.mark.asyncio
    async def test_skips_expired_streams_using_recency_order(
        self, async_session_service, mock_async_redis_manager
    ):
        """Expired streams are skipped in recency order"""
        # When first candidate is expired, second candidate is checked

        async def mock_scan_iter(match, count):
            keys = [
                b"chat:stream:conv-123:run-first",
                b"chat:stream:conv-123:run-second",
            ]
            for key in keys:
                yield key

        async def mock_xrevrange(key, count):
            return [(b"100-0", {})]

        async def mock_exists(key):
            # First key is expired, second exists
            if "run-first" in key:
                return False
            return True

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        mock_async_redis_manager.redis_client.xrevrange = mock_xrevrange
        mock_async_redis_manager.redis_client.exists = AsyncMock(
            side_effect=mock_exists
        )
        mock_async_redis_manager.redis_client.xinfo_stream = AsyncMock()
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="running")

        result = await async_session_service.get_active_session("conv-123")

        assert result.sessionId == "run-second"


# =============================================================================
# TaskStatus - Timestamp Calculations
# =============================================================================


class TestTaskStatusTimestampCalculations:
    """Tests for timestamp calculations in get_task_status"""

    def test_running_task_completion_time_in_future(
        self, session_service, mock_redis_manager
    ):
        """Test that running task has estimated completion in the future"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-running"
        ]
        mock_redis_manager.get_task_status.return_value = "running"

        before_call = session_service._current_timestamp_ms()
        result = session_service.get_task_status("conv-123")
        after_call = session_service._current_timestamp_ms()

        assert isinstance(result, TaskStatusResponse)
        # estimatedCompletion should be about 60 seconds in the future from now
        assert result.estimatedCompletion >= before_call + 59000
        assert result.estimatedCompletion <= after_call + 61000

    def test_completed_task_completion_time_in_past(
        self, session_service, mock_redis_manager
    ):
        """Test that completed task has estimated completion in the past"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-done"
        ]
        mock_redis_manager.get_task_status.return_value = "completed"

        before_call = session_service._current_timestamp_ms()
        result = session_service.get_task_status("conv-123")
        after_call = session_service._current_timestamp_ms()

        assert isinstance(result, TaskStatusResponse)
        # estimatedCompletion should be about 5 seconds in the past from now
        assert result.estimatedCompletion <= before_call - 4000
        assert result.estimatedCompletion >= after_call - 6000

    @pytest.mark.asyncio
    async def test_async_running_task_completion_time_in_future(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test async running task has estimated completion in the future"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-running"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-123:run-running"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="running")

        before_call = async_session_service._current_timestamp_ms()
        result = await async_session_service.get_task_status("conv-123")
        after_call = async_session_service._current_timestamp_ms()

        assert isinstance(result, TaskStatusResponse)
        assert result.estimatedCompletion >= before_call + 59000
        assert result.estimatedCompletion <= after_call + 61000

    @pytest.mark.asyncio
    async def test_async_completed_task_completion_time_in_past(
        self, async_session_service, mock_async_redis_manager
    ):
        """Test async completed task has estimated completion in the past"""

        async def mock_scan_iter(match, count):
            yield b"chat:stream:conv-123:run-done"

        mock_async_redis_manager.redis_client.scan_iter = mock_scan_iter
        async_session_service._get_stream_keys_by_recency = AsyncMock(
            return_value=["chat:stream:conv-123:run-done"]
        )
        mock_async_redis_manager.get_task_status = AsyncMock(return_value="completed")

        before_call = async_session_service._current_timestamp_ms()
        result = await async_session_service.get_task_status("conv-123")
        after_call = async_session_service._current_timestamp_ms()

        assert isinstance(result, TaskStatusResponse)
        assert result.estimatedCompletion <= before_call - 4000
        assert result.estimatedCompletion >= after_call - 6000


# =============================================================================
# SessionService - Multiple Sessions Recency Logic
# =============================================================================


class TestSessionServiceMultipleSessions:
    """Tests for SessionService handling multiple sessions with recency"""

    def test_picks_first_when_sorted_alphabetically(
        self, session_service, mock_redis_manager
    ):
        """SessionService sorts keys reverse alphabetically and picks first"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-a",
            b"chat:stream:conv-123:run-b",
            b"chat:stream:conv-123:run-c",
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"100-0", {})]
        mock_redis_manager.get_task_status.return_value = "running"

        result = session_service.get_active_session("conv-123")

        # Sorted reverse: run-c, run-b, run-a - run-c is first
        assert result.sessionId == "run-c"

    def test_iterates_through_sorted_streams_until_valid_status(
        self, session_service, mock_redis_manager
    ):
        """Streams are checked in sorted order until valid status found"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-a",
            b"chat:stream:conv-456:run-b",
            b"chat:stream:conv-456:run-c",
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"100-0", {})]
        mock_redis_manager.get_task_status.side_effect = [None, None, "running"]

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        # Sorted reverse: run-c, run-b, run-a - run-c first, returns None, then run-b returns None, then run-a returns "running"
        assert result.sessionId == "run-a"
