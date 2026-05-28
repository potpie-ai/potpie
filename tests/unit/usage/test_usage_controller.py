"""Unit tests for UsageController."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.usage.usage_controller import UsageController


pytestmark = pytest.mark.unit


class TestUsageControllerGetUserUsage:
    @pytest.mark.asyncio
    async def test_get_user_usage_returns_usage_data(self):
        mock_session = MagicMock()
        mock_data = {"total_human_messages": 10, "agent_message_counts": {"agent-1": 10}}
        with patch(
            "app.modules.usage.usage_controller.UsageService.get_usage_data",
            new_callable=AsyncMock,
            return_value=mock_data,
        ):
            result = await UsageController.get_user_usage(
                mock_session,
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                user_id="user-1",
            )
            assert result["total_human_messages"] == 10
            assert result["agent_message_counts"] == {"agent-1": 10}
