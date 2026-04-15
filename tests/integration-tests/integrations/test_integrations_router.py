"""
Integration tests for integrations API endpoints.

Uses the shared client fixture with auth override; exercises GET /connected, GET /list,
and GET /{integration_id} with the real IntegrationsService and database.
"""
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestListConnectedIntegrations:
    """Test GET /api/v1/integrations/connected"""

    async def test_connected_returns_200_and_structure(self, client):
        response = await client.get("/api/v1/integrations/connected")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "success"
        assert "count" in data
        assert "connected_integrations" in data
        assert isinstance(data["connected_integrations"], dict)

    async def test_connected_empty_when_no_integrations(self, client):
        response = await client.get("/api/v1/integrations/connected")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["connected_integrations"] == {}


class TestListIntegrations:
    """Test GET /api/v1/integrations/list"""

    async def test_list_returns_200_and_structure(self, client):
        response = await client.get("/api/v1/integrations/list")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "success"
        assert "count" in data
        assert "integrations" in data
        assert isinstance(data["integrations"], dict)

    async def test_list_with_type_filter(self, client):
        response = await client.get("/api/v1/integrations/list?integration_type=sentry")
        assert response.status_code == 200
        data = response.json()
        assert "integrations" in data

    async def test_list_with_org_slug_filter(self, client):
        response = await client.get("/api/v1/integrations/list?org_slug=my-org")
        assert response.status_code == 200
        data = response.json()
        assert "integrations" in data
