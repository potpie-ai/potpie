"""Unit tests for BaseTask."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.celery.tasks.base_task import BaseTask


pytestmark = pytest.mark.unit


class TestBaseTask:
    """Tests for BaseTask."""

    def test_db_property_creates_session(self):
        """db property creates session via SessionLocal when None."""
        with patch("app.celery.tasks.base_task.SessionLocal") as mock_sl:
            mock_session = MagicMock()
            mock_sl.return_value = mock_session

            class Task(BaseTask):
                _db = None
                request = None

            task = Task()
            assert task._db is None
            d = task.db
            assert d == mock_session
            mock_sl.assert_called_once()
            assert task.db is d  # cached

    def test_run_async_runs_coro(self):
        """run_async runs the coroutine via asyncio.run."""
        class Task(BaseTask):
            request = None

        task = Task()
        result = None

        async def simple_coro():
            return 42

        out = task.run_async(simple_coro())
        assert out == 42

    def test_on_success_closes_db(self):
        """on_success closes db and sets to None."""
        mock_db = MagicMock()
        class Task(BaseTask):
            _db = mock_db
            request = None

        task = Task()
        task.on_success(None, "task-id", (), {})

        mock_db.close.assert_called_once()
        assert task._db is None

    def test_on_success_logs_cancelled_when_retval_false(self):
        """on_success logs 'cancelled' when retval is False."""
        class Task(BaseTask):
            _db = None
            request = None

        task = Task()
        with patch("app.celery.tasks.base_task.logger") as mock_log:
            task.on_success(False, "tid", (), {})
            mock_log.info.assert_called_once()
            assert "cancelled" in str(mock_log.info.call_args).lower()

    def test_on_failure_closes_db(self):
        """on_failure closes db and sets to None."""
        mock_db = MagicMock()
        class Task(BaseTask):
            _db = mock_db
            request = None

        task = Task()
        task.on_failure(ValueError("err"), "tid", (), {}, None)

        mock_db.close.assert_called_once()
        assert task._db is None

    def test_on_retry_logs(self):
        """on_retry logs warning."""
        class Task(BaseTask):
            _db = None
            request = None

        task = Task()
        with patch("app.celery.tasks.base_task.logger") as mock_log:
            task.on_retry(ValueError("retry"), "tid", (), {}, None)
            mock_log.warning.assert_called_once()
