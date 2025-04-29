import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from app.modules.auth.auth_service import AuthService


class TestAuthService:
    """Test cases for AuthService class."""

    @pytest.fixture
    def auth_service(self):
        """Create an instance of AuthService for testing."""
        return AuthService()

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_IDENTITY_TOOL_KIT_KEY": "test_key",
                "isDevelopmentMode": "disabled",
                "defaultUsername": "test_user",
            },
        ):
            yield

    @pytest.fixture
    def mock_successful_login_response(self):
        """Mock successful login response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "idToken": "test_token",
            "email": "test@example.com",
            "localId": "test_user_id",
        }
        mock_response.raise_for_status.return_value = None
        return mock_response

    @pytest.fixture
    def mock_failed_login_response(self):
        """Mock failed login response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": {"message": "INVALID_PASSWORD"}}
        mock_response.raise_for_status.side_effect = Exception(mock_response.json())
        return mock_response

    class TestLogin:
        """Test cases for login functionality."""

        def test_login_success(
            self, auth_service, mock_env_vars, mock_successful_login_response
        ):
            """Test successful login with valid credentials."""
            with patch("requests.post", return_value=mock_successful_login_response):
                result = auth_service.login("test@example.com", "valid_password")
                assert result["idToken"] == "test_token"
                assert result["email"] == "test@example.com"

        def test_login_invalid_credentials(
            self, auth_service, mock_env_vars, mock_failed_login_response
        ):
            """Test login with invalid credentials."""
            with patch("requests.post", return_value=mock_failed_login_response):
                with pytest.raises(Exception) as exc_info:
                    auth_service.login("test@example.com", "invalid_password")
                assert "INVALID_PASSWORD" in str(exc_info.value)

        def test_login_empty_credentials(self, auth_service, mock_env_vars):
            """Test login with empty credentials."""
            mock_response = MagicMock()
            mock_response.json.return_value = {"error": {"message": "MISSING_EMAIL"}}
            mock_response.raise_for_status.side_effect = Exception(mock_response.json())

            with patch("requests.post", return_value=mock_response):
                with pytest.raises(Exception) as exc_info:
                    auth_service.login("", "")
                assert "MISSING_EMAIL" in str(exc_info.value)

        def test_login_network_error(self, auth_service, mock_env_vars):
            """Test login with network connection error."""
            with patch(
                "requests.post",
                side_effect=requests.exceptions.ConnectionError("Failed to connect"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.login("test@example.com", "password")
                assert isinstance(exc_info.value, requests.exceptions.ConnectionError)

        def test_login_timeout_error(self, auth_service, mock_env_vars):
            """Test login with request timeout."""
            with patch(
                "requests.post",
                side_effect=requests.exceptions.Timeout("Request timed out"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.login("test@example.com", "password")
                assert isinstance(exc_info.value, requests.exceptions.Timeout)

    class TestSignup:
        """Test cases for signup functionality."""

        @pytest.fixture
        def mock_user(self):
            """Mock user object for successful signup."""
            mock_user = MagicMock()
            mock_user.uid = "test_user_id"
            mock_user.email = "test@example.com"
            mock_user.display_name = "Test User"
            return mock_user

        def test_signup_success(self, auth_service, mock_user):
            """Test successful user signup."""
            with patch("firebase_admin.auth.create_user", return_value=mock_user):
                result = auth_service.signup(
                    "test@example.com", "password123", "Test User"
                )
                assert result.uid == "test_user_id"
                assert result.email == "test@example.com"
                assert result.display_name == "Test User"

        def test_signup_duplicate_email(self, auth_service):
            """Test signup with duplicate email."""
            with patch(
                "firebase_admin.auth.create_user",
                side_effect=Exception("Email already exists"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.signup(
                        "existing@example.com", "password123", "Test User"
                    )
                assert "Email already exists" in str(exc_info.value)

        def test_signup_invalid_email_format(self, auth_service):
            """Test signup with invalid email format."""
            with patch(
                "firebase_admin.auth.create_user",
                side_effect=Exception("INVALID_EMAIL"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.signup("invalid-email", "password123", "Test User")
                assert "INVALID_EMAIL" in str(exc_info.value)

        def test_signup_weak_password(self, auth_service):
            """Test signup with weak password."""
            with patch(
                "firebase_admin.auth.create_user",
                side_effect=Exception("WEAK_PASSWORD"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.signup("test@example.com", "123", "Test User")
                assert "WEAK_PASSWORD" in str(exc_info.value)

        def test_signup_empty_display_name(self, auth_service):
            """Test signup with empty display name."""
            with patch(
                "firebase_admin.auth.create_user",
                side_effect=Exception("INVALID_DISPLAY_NAME"),
            ):
                with pytest.raises(Exception) as exc_info:
                    auth_service.signup("test@example.com", "password123", "")
                assert "INVALID_DISPLAY_NAME" in str(exc_info.value)

    class TestAuthCheck:
        """Test cases for authentication check functionality."""

        @pytest.fixture
        def mock_request_response(self):
            """Mock request and response objects."""
            mock_request = MagicMock()
            mock_response = MagicMock()
            return mock_request, mock_response

        @pytest.mark.asyncio
        async def test_check_auth_valid_token(
            self, auth_service, mock_env_vars, mock_request_response
        ):
            """Test authentication check with valid token."""
            mock_request, mock_response = mock_request_response
            mock_credential = MagicMock()
            mock_credential.credentials = "valid_token"

            mock_decoded_token = {"uid": "test_user_id", "email": "test@example.com"}

            with patch(
                "firebase_admin.auth.verify_id_token", return_value=mock_decoded_token
            ):
                result = await auth_service.check_auth(
                    mock_request, mock_response, mock_credential
                )
                assert result == mock_decoded_token
                assert mock_request.state.user == mock_decoded_token

        @pytest.mark.asyncio
        async def test_check_auth_invalid_token(
            self, auth_service, mock_env_vars, mock_request_response
        ):
            """Test authentication check with invalid token."""
            mock_request, mock_response = mock_request_response
            mock_credential = MagicMock()
            mock_credential.credentials = "invalid_token"

            with patch(
                "firebase_admin.auth.verify_id_token",
                side_effect=Exception("Invalid token"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await auth_service.check_auth(
                        mock_request, mock_response, mock_credential
                    )
                assert exc_info.value.status_code == 401
                assert "Invalid authentication from Firebase" in str(
                    exc_info.value.detail
                )

        @pytest.mark.asyncio
        async def test_check_auth_missing_token(
            self, auth_service, mock_env_vars, mock_request_response
        ):
            """Test authentication check with missing token."""
            mock_request, mock_response = mock_request_response

            with pytest.raises(HTTPException) as exc_info:
                await auth_service.check_auth(mock_request, mock_response, None)
            assert exc_info.value.status_code == 401
            assert "Bearer authentication is needed" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_check_auth_development_mode(
            self, auth_service, mock_request_response
        ):
            """Test authentication check in development mode."""
            mock_request, mock_response = mock_request_response

            with patch.dict(
                os.environ,
                {"isDevelopmentMode": "enabled", "defaultUsername": "dev_user"},
            ):
                result = await auth_service.check_auth(
                    mock_request, mock_response, None
                )
                assert result["user_id"] == "dev_user"
                assert result["email"] == "defaultuser@potpie.ai"
                assert mock_request.state.user == {"user_id": "dev_user"}

        @pytest.mark.asyncio
        async def test_check_auth_expired_token(
            self, auth_service, mock_env_vars, mock_request_response
        ):
            """Test authentication check with expired token."""
            mock_request, mock_response = mock_request_response
            mock_credential = MagicMock()
            mock_credential.credentials = "expired_token"

            with patch(
                "firebase_admin.auth.verify_id_token",
                side_effect=Exception("Token has expired"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await auth_service.check_auth(
                        mock_request, mock_response, mock_credential
                    )
                assert exc_info.value.status_code == 401
                assert "Invalid authentication from Firebase" in str(
                    exc_info.value.detail
                )

        @pytest.mark.asyncio
        async def test_check_auth_malformed_token(
            self, auth_service, mock_env_vars, mock_request_response
        ):
            """Test authentication check with malformed token."""
            mock_request, mock_response = mock_request_response
            mock_credential = MagicMock()
            mock_credential.credentials = "malformed_token"

            with patch(
                "firebase_admin.auth.verify_id_token",
                side_effect=Exception("Malformed token"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await auth_service.check_auth(
                        mock_request, mock_response, mock_credential
                    )
                assert exc_info.value.status_code == 401
                assert "Invalid authentication from Firebase" in str(
                    exc_info.value.detail
                )
