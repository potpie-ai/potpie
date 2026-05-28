"""
Unit tests for Celery agent_tasks.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from app.celery.tasks.agent_tasks import (
    _resolve_user_email_for_celery,
    execute_agent_background,
)
from app.modules.users.user_model import User


pytestmark = pytest.mark.unit


class TestResolveUserEmailForCelery:
    """Tests for _resolve_user_email_for_celery."""

    def test_resolve_user_email_found_via_user_service(self, db_session):
        """User found by UserService returns email."""
        user = User(
            uid="celery-test-user",
            email="celery@example.com",
            display_name="Celery User",
            email_verified=True,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(user)
        db_session.commit()

        with patch(
            "app.celery.tasks.agent_tasks.UserService"
        ) as mock_user_service_class:
            mock_service = MagicMock()
            mock_service.get_user_by_uid.return_value = user
            mock_user_service_class.return_value = mock_service

            result = _resolve_user_email_for_celery(db_session, "celery-test-user")

        assert result == "celery@example.com"

    def test_resolve_user_email_not_found_returns_empty(self, db_session):
        """User not found returns empty string."""
        with patch(
            "app.celery.tasks.agent_tasks.UserService"
        ) as mock_user_service_class:
            mock_service = MagicMock()
            mock_service.get_user_by_uid.return_value = None
            mock_user_service_class.return_value = mock_service

            # Direct query also returns None (no user in db)
            result = _resolve_user_email_for_celery(db_session, "nonexistent-user")

        assert result == ""

    def test_resolve_user_email_fallback_direct_query(self, db_session):
        """When UserService returns None but direct query finds user, return email."""
        user = User(
            uid="direct-query-user",
            email="direct@example.com",
            display_name="Direct",
            email_verified=True,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(user)
        db_session.commit()

        with patch(
            "app.celery.tasks.agent_tasks.UserService"
        ) as mock_user_service_class:
            mock_service = MagicMock()
            mock_service.get_user_by_uid.return_value = None
            mock_user_service_class.return_value = mock_service

            result = _resolve_user_email_for_celery(db_session, "direct-query-user")

        assert result == "direct@example.com"


class TestClearPydanticAiCache:
    """Tests for _clear_pydantic_ai_http_client_cache."""

    def test_clear_cache_no_op_when_not_celery_worker(self):
        """When CELERY_WORKER is not set, cache clear is no-op."""
        from app.celery.tasks.agent_tasks import _clear_pydantic_ai_http_client_cache
        import os

        with patch.dict(os.environ, {"CELERY_WORKER": ""}, clear=False):
            _clear_pydantic_ai_http_client_cache()  # should not raise
