"""Unit tests for FW001 password policy and register-email gate."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.auth.password_policy import (
    PASSWORD_POLICY_ERROR,
    PasswordPolicyError,
    is_password_valid,
    validate_password,
)
from app.modules.auth.auth_service import AuthService
from app.modules.auth.auth_router import auth_router


pytestmark = pytest.mark.unit

STRONG_PASSWORD = "ValidPassword1!"  # 15 chars, meets all classes


class TestPasswordPolicy:
    def test_accepts_strong_password(self):
        assert is_password_valid(STRONG_PASSWORD) is True
        validate_password(STRONG_PASSWORD)  # does not raise

    def test_rejects_short_password(self):
        assert is_password_valid("Short1!") is False
        with pytest.raises(PasswordPolicyError, match="15 or more"):
            validate_password("Short1!")

    def test_rejects_missing_uppercase(self):
        assert is_password_valid("validpassword1!") is False

    def test_rejects_missing_lowercase(self):
        assert is_password_valid("VALIDPASSWORD1!") is False

    def test_rejects_missing_number(self):
        assert is_password_valid("ValidPassword!!") is False

    def test_rejects_missing_special(self):
        # hyphen is not in Firebase special set
        assert is_password_valid("ValidPassword12-") is False
        assert is_password_valid("ValidPassword12_") is True

    def test_error_message_does_not_echo_password(self):
        secret = "weakpass"
        with pytest.raises(PasswordPolicyError) as exc_info:
            validate_password(secret)
        assert secret not in str(exc_info.value)
        assert exc_info.value.message == PASSWORD_POLICY_ERROR


class TestAuthServiceSignupPolicyGate:
    def test_weak_password_never_calls_create_user(self):
        service = AuthService()
        with patch("app.modules.auth.auth_service.auth.create_user") as mock_create:
            result, error = service.signup(
                "user@example.com", "weak", "User"
            )
            mock_create.assert_not_called()
            assert result is None
            assert error["code"] == "weak_password"
            assert error["error"] == PASSWORD_POLICY_ERROR

    def test_strong_password_calls_create_user_once(self):
        service = AuthService()
        mock_user = MagicMock()
        mock_user.uid = "uid-1"
        mock_user.email = "user@example.com"
        with patch(
            "app.modules.auth.auth_service.auth.create_user",
            return_value=mock_user,
        ) as mock_create:
            result, error = service.signup(
                "user@example.com", STRONG_PASSWORD, "User"
            )
            mock_create.assert_called_once()
            assert error is None
            assert result["user"].uid == "uid-1"


class TestRegisterEmailEndpoint:
    def _client(self) -> TestClient:
        app = FastAPI()
        app.include_router(auth_router, prefix="/api/v1")
        return TestClient(app)

    def test_weak_password_returns_400_without_create_user(self):
        client = self._client()
        with patch("app.modules.auth.auth_service.auth.create_user") as mock_create:
            response = client.post(
                "/api/v1/auth/register-email",
                json={
                    "email": "user@example.com",
                    "password": "short",
                    "display_name": "User",
                },
            )
            mock_create.assert_not_called()
        assert response.status_code == 400
        assert response.json()["error"] == PASSWORD_POLICY_ERROR

    def test_strong_password_returns_custom_token(self):
        client = self._client()
        mock_user = MagicMock()
        mock_user.uid = "uid-abc"
        mock_user.email = "user@example.com"
        with patch(
            "app.modules.auth.auth_service.auth.create_user",
            return_value=mock_user,
        ) as mock_create, patch(
            "app.modules.auth.auth_service.auth.create_custom_token",
            return_value=b"custom-token-bytes",
        ):
            response = client.post(
                "/api/v1/auth/register-email",
                json={
                    "email": "user@example.com",
                    "password": STRONG_PASSWORD,
                    "display_name": "User",
                },
            )
            mock_create.assert_called_once()
        assert response.status_code == 201
        body = response.json()
        assert body["uid"] == "uid-abc"
        assert body["customToken"] == "custom-token-bytes"
        assert body["email"] == "user@example.com"
