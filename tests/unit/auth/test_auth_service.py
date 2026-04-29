"""
Unit tests for AuthService (app/modules/auth/auth_service.py)

Tests cover:
- login (sync) with mocked httpx
- login_async with mocked httpx
- signup with mocked firebase_admin
- create_custom_token with mocked firebase_admin
- check_auth in development mode and production mode
"""

import os
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import httpx

from app.modules.auth.auth_service import AuthService, auth_handler


pytestmark = pytest.mark.unit


class TestAuthServiceLogin:
    """Tests for AuthService.login (sync)"""

    def test_login_success(self):
        """Test successful login returns token data"""
        service = AuthService()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "idToken": "mock-id-token",
            "refreshToken": "mock-refresh-token",
            "localId": "user-123",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.modules.auth.auth_service.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = service.login("test@example.com", "password123")

            assert result["idToken"] == "mock-id-token"
            assert result["localId"] == "user-123"
            mock_client.post.assert_called_once()

    def test_login_http_status_error(self):
        """Test login raises HTTPException on auth failure"""
        service = AuthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "INVALID_PASSWORD"}}
        
        http_error = httpx.HTTPStatusError(
            "Auth failed",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("app.modules.auth.auth_service.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value.raise_for_status.side_effect = http_error
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                service.login("test@example.com", "wrongpassword")

            assert exc_info.value.status_code == 401

    def test_login_network_error(self):
        """Test login raises 502 on network error"""
        service = AuthService()

        with patch("app.modules.auth.auth_service.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                service.login("test@example.com", "password123")

            assert exc_info.value.status_code == 502
            assert "Upstream auth request failed" in exc_info.value.detail


class TestAuthServiceLoginAsync:
    """Tests for AuthService.login_async"""

    @pytest.mark.asyncio
    async def test_login_async_success(self):
        """Test successful async login"""
        service = AuthService()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "idToken": "mock-id-token",
            "refreshToken": "mock-refresh-token",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.modules.auth.auth_service.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await service.login_async("test@example.com", "password123")

            assert result["idToken"] == "mock-id-token"

    @pytest.mark.asyncio
    async def test_login_async_http_error(self):
        """Test async login raises HTTPException on failure"""
        service = AuthService()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "EMAIL_NOT_FOUND"}}
        mock_response.text = "Email not found"

        http_error = httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("app.modules.auth.auth_service.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            # Make post return a response that raises on raise_for_status
            mock_post_response = MagicMock()
            mock_post_response.raise_for_status.side_effect = http_error
            mock_client.post = AsyncMock(return_value=mock_post_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await service.login_async("nonexistent@example.com", "password")

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_login_async_network_error(self):
        """Test async login raises 502 on network error"""
        service = AuthService()

        with patch("app.modules.auth.auth_service.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await service.login_async("test@example.com", "password")

            assert exc_info.value.status_code == 502


class TestAuthServiceSignup:
    """Tests for AuthService.signup"""

    def test_signup_success(self):
        """Test successful user signup"""
        service = AuthService()
        mock_user = MagicMock()
        mock_user.uid = "new-user-123"
        mock_user.email = "newuser@example.com"

        with patch("app.modules.auth.auth_service.auth.create_user", return_value=mock_user):
            result, error = service.signup("newuser@example.com", "securepass", "New User")

            assert error is None
            assert result["user"].uid == "new-user-123"
            assert "successfully" in result["message"]

    def test_signup_firebase_error(self):
        """Test signup handles Firebase errors"""
        service = AuthService()

        # Create a custom exception class that mimics FirebaseError
        class MockFirebaseError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)

        mock_error = MockFirebaseError("Email already exists")

        with patch("app.modules.auth.auth_service.auth.create_user") as mock_create:
            # Patch FirebaseError to be our mock class for the isinstance check
            with patch(
                "app.modules.auth.auth_service.FirebaseError", MockFirebaseError
            ):
                mock_create.side_effect = mock_error

                result, error = service.signup("existing@example.com", "pass", "User")

                assert result is None
                assert "Firebase error" in error["error"]
                assert "Email already exists" in error["error"]

    def test_signup_value_error(self):
        """Test signup handles invalid input"""
        service = AuthService()

        with patch("app.modules.auth.auth_service.auth.create_user") as mock_create:
            mock_create.side_effect = ValueError("Invalid email format")

            result, error = service.signup("invalid-email", "pass", "User")

            assert result is None
            assert "Invalid input" in error["error"]

    def test_signup_unexpected_error(self):
        """Test signup handles unexpected errors"""
        service = AuthService()

        with patch("app.modules.auth.auth_service.auth.create_user") as mock_create:
            mock_create.side_effect = RuntimeError("Unexpected failure")

            result, error = service.signup("test@example.com", "pass", "User")

            assert result is None
            assert "unexpected error" in error["error"]


class TestAuthServiceCreateCustomToken:
    """Tests for AuthService.create_custom_token"""

    def test_create_custom_token_success_bytes(self):
        """Test creating custom token (bytes response)"""
        with patch("app.modules.auth.auth_service.auth.create_custom_token") as mock_create:
            mock_create.return_value = b"custom-token-bytes"

            token = AuthService.create_custom_token("user-123")

            assert token == "custom-token-bytes"
            mock_create.assert_called_once_with("user-123")

    def test_create_custom_token_success_string(self):
        """Test creating custom token (string response)"""
        with patch("app.modules.auth.auth_service.auth.create_custom_token") as mock_create:
            mock_create.return_value = "custom-token-string"

            token = AuthService.create_custom_token("user-456")

            assert token == "custom-token-string"

    def test_create_custom_token_error(self):
        """Test custom token returns None on error"""
        with patch("app.modules.auth.auth_service.auth.create_custom_token") as mock_create:
            mock_create.side_effect = Exception("Token creation failed")

            token = AuthService.create_custom_token("user-789")

            assert token is None


class TestAuthServiceCheckAuth:
    """Tests for AuthService.check_auth"""

    @pytest.mark.asyncio
    async def test_check_auth_development_mode_no_credential(self):
        """Test check_auth in development mode without credentials"""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None
        mock_request.state = MagicMock()

        with patch.dict(os.environ, {"isDevelopmentMode": "enabled", "defaultUsername": "dev-user"}):
            result = await AuthService.check_auth(mock_request, None, None)

            assert result["user_id"] == "dev-user"
            assert result["email"] == "defaultuser@potpie.ai"

    @pytest.mark.asyncio
    async def test_check_auth_no_credential_not_dev_mode(self):
        """Test check_auth raises 401 when no credential and not in dev mode"""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None

        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                await AuthService.check_auth(mock_request, None, None)

            assert exc_info.value.status_code == 401
            assert "Bearer authentication" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_check_auth_valid_token(self):
        """Test check_auth with valid Firebase token"""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer valid-token"
        mock_request.state = MagicMock()
        mock_response = MagicMock()

        decoded_token = {
            "uid": "firebase-user-123",
            "email": "user@example.com",
        }

        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            with patch("app.modules.auth.auth_service.auth.verify_id_token", return_value=decoded_token):
                credential = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
                result = await AuthService.check_auth(mock_request, mock_response, credential)

                assert result["uid"] == "firebase-user-123"
                assert result["user_id"] == "firebase-user-123"
                assert result["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_check_auth_does_not_log_token_contents(self):
        """Auth checks should not emit bearer token contents to logs."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer sensitive-token"
        mock_request.state = MagicMock()
        mock_response = MagicMock()
        decoded_token = {"uid": "firebase-user-123", "email": "user@example.com"}

        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            with patch(
                "app.modules.auth.auth_service.auth.verify_id_token",
                return_value=decoded_token,
            ):
                with patch("app.modules.auth.auth_service.logging") as mock_logging:
                    credential = HTTPAuthorizationCredentials(
                        scheme="Bearer", credentials="sensitive-token"
                    )
                    await AuthService.check_auth(
                        mock_request, mock_response, credential
                    )

        assert not any(
            "sensitive-token" in str(value)
            for call in mock_logging.mock_calls
            for value in [*call.args, *call.kwargs.values()]
        )

    @pytest.mark.asyncio
    async def test_check_auth_invalid_token(self):
        """Test check_auth raises 401 on invalid token"""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer invalid-token"

        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            with patch("app.modules.auth.auth_service.auth.verify_id_token") as mock_verify:
                mock_verify.side_effect = Exception("Token expired")
                credential = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")

                with pytest.raises(HTTPException) as exc_info:
                    await AuthService.check_auth(mock_request, None, credential)

                assert exc_info.value.status_code == 401
                assert "Invalid authentication" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_check_auth_manual_bearer_extraction(self):
        """Test check_auth extracts Bearer token from header when credential is not HTTPAuthorizationCredentials"""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer manual-token"
        mock_request.state = MagicMock()
        mock_response = MagicMock()

        decoded_token = {"uid": "manual-user", "email": "manual@example.com"}

        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            with patch("app.modules.auth.auth_service.auth.verify_id_token", return_value=decoded_token):
                # Pass a non-HTTPAuthorizationCredentials object (simulates Depends() default)
                result = await AuthService.check_auth(mock_request, mock_response, "not-a-credential")

                assert result["uid"] == "manual-user"
                assert result["user_id"] == "manual-user"


class TestAuthHandler:
    """Test the module-level auth_handler instance"""

    def test_auth_handler_is_auth_service(self):
        """Test auth_handler is an AuthService instance"""
        assert isinstance(auth_handler, AuthService)
