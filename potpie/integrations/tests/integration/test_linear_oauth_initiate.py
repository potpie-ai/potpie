"""Integration tests for POST /api/v1/integrations/linear/initiate."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from integrations.adapters.inbound.http.integrations_router import _verify_oauth_state

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestLinearInitiate:
    """POST /api/v1/integrations/linear/initiate — server-side identity."""

    async def test_initiate_returns_authorization_url_signed_with_test_user(
        self, client
    ):
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
        signed_state = qs.get("state", [""])[0]
        assert signed_state, "state must be present"
        assert _verify_oauth_state(signed_state) == "test-user"
        assert qs.get("prompt") == ["consent"]

    async def test_initiate_rejects_mismatched_redirect_uri(self, client):
        resp = await client.post(
            "/api/v1/integrations/linear/initiate",
            json={"redirect_uri": "http://attacker.example.com/steal"},
        )
        assert resp.status_code == 400
        assert "redirect_uri" in resp.text

    async def test_initiate_state_is_not_taken_from_body(self, client):
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
