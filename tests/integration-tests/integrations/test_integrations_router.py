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


class TestLinearInitiate:
    """POST /api/v1/integrations/linear/initiate — server-side identity."""

    async def test_initiate_returns_authorization_url_signed_with_test_user(
        self, client
    ):
        """Happy path: server reads test-user from the auth override and
        produces a Linear authorization URL whose ``state`` round-trips
        back to ``test-user`` via ``_verify_oauth_state``."""
        from urllib.parse import parse_qs, urlparse

        from integrations.adapters.inbound.http.integrations_router import (
            _verify_oauth_state,
        )

        body = {
            "redirect_uri": "http://test/api/v1/integrations/linear/callback",
        }
        resp = await client.post("/api/v1/integrations/linear/initiate", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "success"
        auth_url = data["authorization_url"]
        assert auth_url.startswith("https://linear.app/oauth/authorize")

        parsed = urlparse(auth_url)
        qs = parse_qs(parsed.query)
        assert qs.get("client_id"), "Linear client_id should be in the auth URL"
        assert qs.get("redirect_uri") == [body["redirect_uri"]]
        # The state should decode back to the authenticated user id.
        signed_state = qs.get("state", [""])[0]
        assert signed_state, "state must be present"
        recovered = _verify_oauth_state(signed_state)
        assert recovered == "test-user"
        # prompt=consent ensures the workspace picker is shown every time
        # rather than silently reusing the user's primary workspace.
        assert qs.get("prompt") == ["consent"]

    async def test_initiate_rejects_mismatched_redirect_uri(self, client):
        """Confused-deputy guard: redirect_uri must point at our callback path."""
        resp = await client.post(
            "/api/v1/integrations/linear/initiate",
            json={"redirect_uri": "http://attacker.example.com/steal"},
        )
        assert resp.status_code == 400
        assert "redirect_uri" in resp.text

    async def test_initiate_state_is_not_taken_from_body(self, client):
        """Even if the caller passes a state, the server must ignore it
        and sign only the authenticated user_id."""
        from urllib.parse import parse_qs, urlparse

        from integrations.adapters.inbound.http.integrations_router import (
            _verify_oauth_state,
        )

        resp = await client.post(
            "/api/v1/integrations/linear/initiate",
            json={
                "redirect_uri": "http://test/api/v1/integrations/linear/callback",
                "state": "attacker-user-id",
            },
        )
        assert resp.status_code == 200
        auth_url = resp.json()["authorization_url"]
        signed_state = parse_qs(urlparse(auth_url).query).get("state", [""])[0]
        assert _verify_oauth_state(signed_state) == "test-user"
        assert _verify_oauth_state(signed_state) != "attacker-user-id"
