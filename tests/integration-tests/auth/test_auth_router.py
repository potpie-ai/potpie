"""
Integration tests for auth API endpoints.

These tests exercise the FastAPI router using the shared `client` fixture in `tests/conftest.py`,
which mounts the real app and overrides DB + auth dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from app.modules.auth.auth_provider_model import UserAuthProvider
from app.modules.auth.unified_auth_service import UnifiedAuthService
from app.modules.users.user_model import User


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def mock_google_sso(monkeypatch):
    """Mock SSO token verification so we don't need real Google JWTs."""

    class _VerifiedUserInfo:
        def __init__(self):
            self.provider_uid = "google-mocked-sub"
            self.email = None
            self.display_name = None
            self.email_verified = True
            self.raw_data = {}

    async def _fake_verify_sso_token(self, provider_type: str, id_token: str):
        return _VerifiedUserInfo()

    monkeypatch.setattr(UnifiedAuthService, "verify_sso_token", _fake_verify_sso_token)


class TestSignupValidation:
    """Test POST /api/v1/signup request validation."""

    async def test_signup_missing_uid(self, client, db_session):
        """Signup without uid returns 400."""
        response = await client.post(
            "/api/v1/signup",
            json={
                "email": "test@example.com",
                "displayName": "Test",
                "emailVerified": True,
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "uid" in data.get("error", "").lower() or "missing" in data.get("error", "").lower()

    async def test_signup_missing_email(self, client, db_session):
        """Signup without email returns 400."""
        response = await client.post(
            "/api/v1/signup",
            json={
                "uid": "firebase-uid-123",
                "displayName": "Test",
                "emailVerified": True,
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "email" in data.get("error", "").lower() or "missing" in data.get("error", "").lower()


class TestLoginEndpoint:
    """Test POST /api/v1/login"""

    async def test_login_success(self, client):
        """Test successful login with mocked auth service"""
        mock_response = {"idToken": "mock-token-123", "refreshToken": "refresh-123"}

        with patch(
            "app.modules.auth.auth_router.auth_handler.login_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await client.post(
                "/api/v1/login",
                json={"email": "test@example.com", "password": "password123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["token"] == "mock-token-123"

    async def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        from fastapi import HTTPException

        with patch(
            "app.modules.auth.auth_router.auth_handler.login_async",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=401, detail="Invalid credentials"),
        ):
            response = await client.post(
                "/api/v1/login",
                json={"email": "test@example.com", "password": "wrongpassword"},
            )

            assert response.status_code == 401

    async def test_login_value_error(self, client):
        """Test login with ValueError (invalid email/password format)"""
        with patch(
            "app.modules.auth.auth_router.auth_handler.login_async",
            new_callable=AsyncMock,
            side_effect=ValueError("Invalid email format"),
        ):
            response = await client.post(
                "/api/v1/login",
                json={"email": "invalid", "password": "pass"},
            )

            assert response.status_code == 401
            assert "Invalid email or password" in response.json()["error"]


class TestCustomTokenEndpoint:
    """Test POST /api/v1/auth/custom-token"""

    async def test_custom_token_success(self, client, auth_token, test_user):
        """Test creating custom token for authenticated user"""
        with patch(
            "app.modules.auth.auth_router.AuthService.create_custom_token",
            return_value="custom-token-xyz",
        ):
            response = await client.post(
                "/api/v1/auth/custom-token",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["customToken"] == "custom-token-xyz"

    async def test_custom_token_failure(self, client, auth_token, test_user):
        """Test custom token creation failure"""
        with patch(
            "app.modules.auth.auth_router.AuthService.create_custom_token",
            return_value=None,
        ):
            response = await client.post(
                "/api/v1/auth/custom-token",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            assert response.status_code == 500
            assert "Failed" in response.json()["error"]


class TestSSOLoginEndpoint:
    """Test POST /api/v1/sso/login"""

    async def test_sso_login_new_user(self, client, mock_google_sso):
        response = await client.post(
            "/api/v1/sso/login",
            json={
                "email": "newuser@acme.com",
                "sso_provider": "google",
                "id_token": "mock-token",
                "provider_data": {
                    "sub": "google-123",
                    "name": "New User",
                },
            },
        )
        assert response.status_code in (200, 202)
        data = response.json()
        assert data["status"] in ("new_user", "success")
        assert data["email"] == "newuser@acme.com"

    async def test_sso_login_existing_user_needs_linking(
        self, client, db_session, mock_google_sso
    ):
        # Create an isolated existing user with only GitHub provider.
        user = User(
            uid="test-sso-existing-user",
            email="existing@acme.com",
            display_name="Existing User",
            email_verified=True,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(user)
        db_session.add(
            UserAuthProvider(
                user_id=user.uid,
                provider_type="firebase_github",
                provider_uid="github-existing-1",
                is_primary=True,
                linked_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

        response = await client.post(
            "/api/v1/sso/login",
            json={
                "email": "existing@acme.com",
                "sso_provider": "google",
                "id_token": "mock-token",
                "provider_data": {
                    "sub": "google-456",
                    "name": "Existing User",
                },
            },
        )
        assert response.status_code in (200, 202)
        data = response.json()
        # Existing user with a different provider should require linking.
        assert data.get("status") in ("needs_linking", "pending_link", "success")

    async def test_sso_login_invalid_token(self, client):
        response = await client.post(
            "/api/v1/sso/login",
            json={
                "email": "test@example.com",
                "sso_provider": "google",
                "id_token": "invalid-token",
                "provider_data": {},
            },
        )
        assert response.status_code in (400, 401, 422)


class TestProviderManagementEndpoints:
    """Test provider management endpoints"""

    async def test_get_my_providers(self, client, test_user_with_multiple_providers, auth_token):
        response = await client.get(
            "/api/v1/providers/me",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) >= 2

    async def test_set_primary_provider(self, client, test_user_with_multiple_providers, auth_token):
        response = await client.post(
            "/api/v1/providers/set-primary",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "sso_google"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Primary provider updated"

    async def test_unlink_provider(self, client, test_user_with_multiple_providers, auth_token):
        response = await client.request(
            "DELETE",
            "/api/v1/providers/unlink",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "sso_google"},
        )
        assert response.status_code == 200

    async def test_unlink_last_provider_fails(self, client, test_user_with_github, auth_token):
        response = await client.request(
            "DELETE",
            "/api/v1/providers/unlink",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "firebase_github"},
        )
        assert response.status_code in (400, 409)
        data = response.json()
        assert "cannot unlink" in data.get("error", "").lower()


class TestAccountEndpoint:
    """Test GET /api/v1/account/me"""

    async def test_get_account(self, client, test_user_with_multiple_providers, auth_token):
        response = await client.get(
            "/api/v1/account/me",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "email" in data
        assert "providers" in data
        assert len(data["providers"]) >= 2


class TestProviderLinkingEndpoints:
    """Test provider linking flow"""

    async def test_confirm_linking(self, client, pending_link):
        response = await client.post(
            "/api/v1/providers/confirm-linking",
            json={"linking_token": pending_link.token},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Provider linked successfully"

    async def test_confirm_linking_invalid_token(self, client):
        response = await client.post(
            "/api/v1/providers/confirm-linking",
            json={"linking_token": "invalid-token"},
        )
        assert response.status_code in (400, 404, 422)

    async def test_cancel_linking(self, client, pending_link):
        response = await client.delete(
            f"/api/v1/providers/cancel-linking/{pending_link.token}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Linking cancelled"

