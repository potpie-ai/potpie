"""Integration tests for usage API."""
from datetime import datetime, timedelta, timezone

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestGetUsage:
    async def test_get_usage_returns_200_and_structure(self, client):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        response = await client.get(
            "/api/v1/usage/usage",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_human_messages" in data
        assert "agent_message_counts" in data
        assert isinstance(data["agent_message_counts"], dict)
