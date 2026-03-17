"""Unit tests for UsageService."""
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.modules.usage.usage_service import UsageService


pytestmark = pytest.mark.unit


class TestUsageServiceGetUsageData:
    @pytest.mark.asyncio
    async def test_get_usage_data_returns_structure(self):
        mock_result = MagicMock()
        mock_result.all.return_value = [("agent-1", 5), ("agent-2", 3)]
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        result = await UsageService.get_usage_data(
            mock_session,
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            user_id="user-1",
        )
        assert "total_human_messages" in result
        assert "agent_message_counts" in result
        assert result["total_human_messages"] == 8
        assert result["agent_message_counts"] == {"agent-1": 5, "agent-2": 3}

    @pytest.mark.asyncio
    async def test_get_usage_data_empty(self):
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        result = await UsageService.get_usage_data(
            mock_session,
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            user_id="user-1",
        )
        assert result["total_human_messages"] == 0
        assert result["agent_message_counts"] == {}

    @pytest.mark.asyncio
    async def test_get_usage_data_sqlalchemy_error_raises(self):
        from sqlalchemy.exc import SQLAlchemyError
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("db error"))
        with pytest.raises(Exception, match="Failed to fetch usage data"):
            await UsageService.get_usage_data(
                mock_session,
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                user_id="user-1",
            )


class TestUsageServiceCheckUsageLimit:
    @pytest.mark.asyncio
    async def test_check_usage_limit_no_subscription_url_returns_true(self):
        with patch.dict(os.environ, {"SUBSCRIPTION_BASE_URL": ""}, clear=False):
            result = await UsageService.check_usage_limit("user-1", AsyncMock())
            assert result is True

    @pytest.mark.asyncio
    async def test_check_usage_limit_with_subscription_under_limit(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"plan_type": "free", "end_date": None}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch.dict(os.environ, {"SUBSCRIPTION_BASE_URL": "https://sub.example.com"}, clear=False):
            with patch("app.modules.usage.usage_service.httpx.AsyncClient", return_value=mock_client):
                with patch.object(
                    UsageService,
                    "get_usage_data",
                    new_callable=AsyncMock,
                    return_value={"total_human_messages": 10, "agent_message_counts": {}},
                ):
                    result = await UsageService.check_usage_limit("user-1", AsyncMock())
                    assert result is True

    @pytest.mark.asyncio
    async def test_check_usage_limit_free_plan_over_limit_raises(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"plan_type": "free", "end_date": None}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        with patch.dict(os.environ, {"SUBSCRIPTION_BASE_URL": "https://sub.example.com"}, clear=False):
            with patch("app.modules.usage.usage_service.httpx.AsyncClient", return_value=mock_client):
                with patch.object(
                    UsageService,
                    "get_usage_data",
                    new_callable=AsyncMock,
                    return_value={"total_human_messages": 50, "agent_message_counts": {}},
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        await UsageService.check_usage_limit("user-1", AsyncMock())
                    assert exc_info.value.status_code == 402
                    assert "limit" in (exc_info.value.detail or "").lower()

