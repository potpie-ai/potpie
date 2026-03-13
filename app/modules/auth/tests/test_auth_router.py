"""
Integration tests for auth API endpoints.

These tests use the full FastAPI app with DB and auth overrides from conftest.
SSO login tests mock the UnifiedAuthService; other endpoints use the check_auth override.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.modules.auth.auth_provider_model import UserAuthProvider


class TestSSOLoginEndpoint:
    """Test POST /api/v1/sso/login
    
    These tests mock UnifiedAuthService entirely to avoid Firebase/Google calls.
    """

    def test_sso_login_new_user(self, client, db_session):
        """Test SSO login for new user creates account"""
        # Mock user object
        mock_user = MagicMock()
        mock_user.uid = "new-uid"
        mock_user.email = "newuser@example.com"

        # Mock response object (authenticate_or_create returns (user, response) tuple)
        # The router calls response.model_dump() so we need to provide a proper dict
        mock_response = MagicMock()
        mock_response.status = "new_user"
        mock_response.needs_linking = False
        mock_response.needs_github_linking = False
        mock_response.existing_providers = []
        mock_response.linking_token = None
        mock_response.model_dump.return_value = {
            "status": "new_user",
            "needs_linking": False,
            "needs_github_linking": False,
            "existing_providers": [],
            "linking_token": None,
        }

        with patch(
            "app.modules.auth.auth_router.UnifiedAuthService"
        ) as mock_unified_auth_class:
            mock_service = mock_unified_auth_class.return_value
            # verify_sso_token is async
            mock_service.verify_sso_token = AsyncMock(
                return_value=MagicMock(
                    email="newuser@example.com",
                    email_verified=True,
                    provider_uid="google-123",
                    display_name="New User",
                    provider_data={"sub": "google-123"},
                )
            )
            # authenticate_or_create returns (user, response) tuple
            mock_service.authenticate_or_create = AsyncMock(
                return_value=(mock_user, mock_response)
            )

            response = client.post(
                "/api/v1/sso/login",
                json={
                    "email": "newuser@example.com",
                    "sso_provider": "google",
                    "id_token": "mock-token",
                    "provider_data": {"sub": "google-123", "name": "New User"},
                },
            )

            # 202 Accepted for new_user (not "success")
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "new_user"

    def test_sso_login_existing_user(self, client, db_session, test_user_with_github):
        """Test SSO login for existing user with different provider triggers linking"""
        mock_user = MagicMock()
        mock_user.uid = test_user_with_github.uid
        mock_user.email = "test@example.com"

        mock_response = MagicMock()
        mock_response.status = "needs_linking"
        mock_response.needs_linking = True
        mock_response.needs_github_linking = False
        mock_response.existing_providers = ["firebase_github"]
        mock_response.linking_token = "mock-linking-token"
        mock_response.model_dump.return_value = {
            "status": "needs_linking",
            "needs_linking": True,
            "needs_github_linking": False,
            "existing_providers": ["firebase_github"],
            "linking_token": "mock-linking-token",
        }

        with patch(
            "app.modules.auth.auth_router.UnifiedAuthService"
        ) as mock_unified_auth_class:
            mock_service = mock_unified_auth_class.return_value
            mock_service.verify_sso_token = AsyncMock(
                return_value=MagicMock(
                    email="test@example.com",
                    email_verified=True,
                    provider_uid="google-456",
                    display_name="Test User",
                    provider_data={"sub": "google-456"},
                )
            )
            mock_service.authenticate_or_create = AsyncMock(
                return_value=(mock_user, mock_response)
            )

            response = client.post(
                "/api/v1/sso/login",
                json={
                    "email": "test@example.com",
                    "sso_provider": "google",
                    "id_token": "mock-token",
                    "provider_data": {"sub": "google-456", "name": "Test User"},
                },
            )

            assert response.status_code in [200, 202]

    def test_sso_login_invalid_token(self, client, db_session):
        """Test SSO login with invalid token returns error"""
        with patch(
            "app.modules.auth.auth_router.UnifiedAuthService"
        ) as mock_unified_auth_class:
            mock_service = mock_unified_auth_class.return_value
            mock_service.verify_sso_token = AsyncMock(
                side_effect=ValueError("Invalid token")
            )

            response = client.post(
                "/api/v1/sso/login",
                json={
                    "email": "test@example.com",
                    "sso_provider": "google",
                    "id_token": "invalid-token",
                    "provider_data": {},
                },
            )

            assert response.status_code == 400


class TestProviderManagementEndpoints:
    """Test provider management endpoints"""

    def test_get_my_providers(self, client, test_user_with_multiple_providers, auth_token):
        """Test GET /api/v1/providers/me"""
        response = client.get(
            "/api/v1/providers/me",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) == 2

    def test_set_primary_provider(self, client, test_user_with_multiple_providers, auth_token):
        """Test POST /api/v1/providers/set-primary"""
        response = client.post(
            "/api/v1/providers/set-primary",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "sso_google"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Primary provider updated"

    def test_unlink_provider(self, client, test_user_with_multiple_providers, auth_token):
        """Test DELETE /api/v1/providers/unlink (uses request() for DELETE with body)"""
        response = client.request(
            method="DELETE",
            url="/api/v1/providers/unlink",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "sso_google"},
        )

        assert response.status_code == 200

    def test_unlink_last_provider_fails(self, client, test_user_with_github, auth_token):
        """Test unlinking last provider returns error"""
        response = client.request(
            method="DELETE",
            url="/api/v1/providers/unlink",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"provider_type": "firebase_github"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "cannot unlink" in data["error"].lower()


class TestAccountEndpoint:
    """Test GET /api/v1/account/me"""

    def test_get_account(self, client, test_user_with_multiple_providers, auth_token):
        """Test getting complete account info"""
        response = client.get(
            "/api/v1/account/me",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "email" in data
        assert "providers" in data
        assert len(data["providers"]) == 2


class TestProviderLinkingEndpoints:
    """Test provider linking flow"""

    def test_confirm_linking(self, client, pending_link):
        """Test POST /api/v1/providers/confirm-linking"""
        response = client.post(
            "/api/v1/providers/confirm-linking",
            json={"linking_token": pending_link.token},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Provider linked successfully"

    def test_confirm_linking_invalid_token(self, client):
        """Test confirming with invalid token"""
        response = client.post(
            "/api/v1/providers/confirm-linking",
            json={"linking_token": "invalid-token"},
        )

        assert response.status_code == 400

    def test_cancel_linking(self, client, pending_link):
        """Test DELETE /api/v1/providers/cancel-linking/{token}"""
        response = client.delete(
            f"/api/v1/providers/cancel-linking/{pending_link.token}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Linking cancelled"
