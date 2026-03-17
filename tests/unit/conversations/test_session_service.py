"""
Unit tests for SessionService (app/modules/conversations/session/session_service.py)

Tests cover:
- get_active_session
- get_task_status
- timestamp utilities
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import redis

from app.modules.conversations.session.session_service import SessionService
from app.modules.conversations.conversation.conversation_schema import (
    ActiveSessionResponse,
    ActiveSessionErrorResponse,
    TaskStatusResponse,
    TaskStatusErrorResponse,
)


pytestmark = pytest.mark.unit


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


class TestCurrentTimestamp:
    """Tests for timestamp utility"""

    def test_current_timestamp_ms(self, session_service):
        """Test timestamp is in milliseconds"""
        ts = session_service._current_timestamp_ms()
        
        # Should be a large number (milliseconds since epoch)
        assert ts > 1000000000000
        # Should be an integer
        assert isinstance(ts, int)


class TestGetActiveSession:
    """Tests for get_active_session"""

    def test_no_active_session(self, session_service, mock_redis_manager):
        """Test returns error when no streams found"""
        mock_redis_manager.redis_client.keys.return_value = []

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert result.conversationId == "conv-123"
        assert "No active session" in result.error

    def test_active_session_found_running(self, session_service, mock_redis_manager):
        """Test returns session info when stream is active"""
        # Mock Redis responses
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-abc"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"123-0", {})]
        mock_redis_manager.get_task_status.return_value = "running"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.conversationId == "conv-123"
        assert result.sessionId == "run-abc"
        assert result.status == "active"
        assert result.cursor == "123-0"

    def test_active_session_completed(self, session_service, mock_redis_manager):
        """Test returns completed status when task is done"""
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

    def test_active_session_idle(self, session_service, mock_redis_manager):
        """Test returns idle status for unknown task status"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-idle"
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = []
        mock_redis_manager.get_task_status.return_value = "unknown"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.status == "idle"
        assert result.cursor == "0-0"

    def test_active_session_stream_not_exists(self, session_service, mock_redis_manager):
        """Test returns error when stream no longer exists"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-old"
        ]
        mock_redis_manager.redis_client.exists.return_value = False

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionErrorResponse)
        assert "No active session" in result.error

    def test_active_session_redis_error(self, session_service, mock_redis_manager):
        """Test raises on Redis connection error"""
        mock_redis_manager.redis_client.keys.side_effect = redis.exceptions.ConnectionError(
            "Connection refused"
        )

        with pytest.raises(redis.exceptions.ConnectionError):
            session_service.get_active_session("conv-123")

    def test_active_session_sorts_streams(self, session_service, mock_redis_manager):
        """Test that multiple streams are sorted to get most recent"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-1",
            b"chat:stream:conv-123:run-3",
            b"chat:stream:conv-123:run-2",
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = [(b"999-0", {})]
        mock_redis_manager.get_task_status.return_value = "running"

        result = session_service.get_active_session("conv-123")

        # Should pick run-3 (highest when sorted in reverse)
        assert result.sessionId == "run-3"


class TestGetTaskStatus:
    """Tests for get_task_status"""

    def test_no_background_task(self, session_service, mock_redis_manager):
        """Test returns error when no streams found"""
        mock_redis_manager.redis_client.keys.return_value = []

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusErrorResponse)
        assert result.conversationId == "conv-456"
        assert "No background task" in result.error

    def test_task_status_running(self, session_service, mock_redis_manager):
        """Test returns active status for running task"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-task"
        ]
        mock_redis_manager.get_task_status.return_value = "running"

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is True
        assert result.sessionId == "run-task"
        assert result.conversationId == "conv-456"
        # Estimated completion should be in the future
        assert result.estimatedCompletion > session_service._current_timestamp_ms() - 1000

    def test_task_status_pending(self, session_service, mock_redis_manager):
        """Test returns active status for pending task"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-pending"
        ]
        mock_redis_manager.get_task_status.return_value = "pending"

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is True

    def test_task_status_completed(self, session_service, mock_redis_manager):
        """Test returns inactive status for completed task"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-done"
        ]
        mock_redis_manager.get_task_status.return_value = "completed"

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.isActive is False
        # Estimated completion should be in the past
        assert result.estimatedCompletion < session_service._current_timestamp_ms() + 1000

    def test_task_status_no_valid_status(self, session_service, mock_redis_manager):
        """Test returns error when no valid task status found"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-1"
        ]
        mock_redis_manager.get_task_status.return_value = None

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusErrorResponse)
        assert "No background task" in result.error

    def test_task_status_tries_multiple_streams(self, session_service, mock_redis_manager):
        """Test iterates through streams to find valid task status"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-456:run-1",
            b"chat:stream:conv-456:run-2",
        ]
        # First stream has no status, second has running
        mock_redis_manager.get_task_status.side_effect = [None, "running"]

        result = session_service.get_task_status("conv-456")

        assert isinstance(result, TaskStatusResponse)
        assert result.sessionId == "run-1"  # First in sorted order (reverse)


class TestStringKeyHandling:
    """Tests for handling string vs bytes keys"""

    def test_handles_string_keys(self, session_service, mock_redis_manager):
        """Test handles string keys (non-bytes)"""
        mock_redis_manager.redis_client.keys.return_value = [
            "chat:stream:conv-123:run-str"  # String, not bytes
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = []
        mock_redis_manager.get_task_status.return_value = "idle"

        result = session_service.get_active_session("conv-123")

        assert isinstance(result, ActiveSessionResponse)
        assert result.sessionId == "run-str"

    def test_handles_mixed_keys(self, session_service, mock_redis_manager):
        """Test handles mix of bytes and string keys"""
        mock_redis_manager.redis_client.keys.return_value = [
            b"chat:stream:conv-123:run-bytes",
            "chat:stream:conv-123:run-string",
        ]
        mock_redis_manager.redis_client.exists.return_value = True
        mock_redis_manager.redis_client.xinfo_stream.return_value = {}
        mock_redis_manager.redis_client.xrevrange.return_value = []
        mock_redis_manager.get_task_status.return_value = "idle"

        result = session_service.get_active_session("conv-123")

        # Should work without error
        assert isinstance(result, ActiveSessionResponse)
