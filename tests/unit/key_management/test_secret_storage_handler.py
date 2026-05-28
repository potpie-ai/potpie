"""Unit tests for SecretStorageHandler (GCP availability, encryption key, format_secret_id, get/store)."""
import os
from unittest.mock import patch, MagicMock

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException

from app.modules.key_management.secret_manager import SecretStorageHandler


pytestmark = pytest.mark.unit


class TestSecretStorageHandlerGcpAvailability:
    def test_gcp_disabled_returns_none(self):
        # Reset cache so this test sees the env
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None
        with patch.dict(os.environ, {"GCP_SECRET_MANAGER_DISABLED": "true"}, clear=False):
            client, project_id = SecretStorageHandler._check_gcp_availability_once()
            assert client is None
            assert project_id is None
            assert SecretStorageHandler._gcp_available is False
        # Reset for other tests
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

    def test_no_gcp_project_returns_none(self):
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None
        with patch.dict(os.environ, {"GCP_PROJECT": ""}, clear=False):
            with patch.dict(os.environ, {"GCP_SECRET_MANAGER_DISABLED": "false"}, clear=False):
                client, project_id = SecretStorageHandler._check_gcp_availability_once()
                assert client is None
                assert project_id is None
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None


class TestSecretStorageHandlerGetEncryptionKey:
    def test_no_key_raises(self):
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": ""}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.get_encryption_key()
            assert exc_info.value.status_code == 500
            assert "SECRET_ENCRYPTION_KEY" in (exc_info.value.detail or "")

    def test_valid_key_returns_fernet(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            f = SecretStorageHandler.get_encryption_key()
            assert f is not None
            assert isinstance(f, Fernet)


class TestSecretStorageHandlerFormatSecretId:
    def test_dev_mode_returns_none(self):
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            assert SecretStorageHandler.format_secret_id("openai", "user-1") is None

    def test_prod_ai_provider_format(self):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            out = SecretStorageHandler.format_secret_id("openai", "user-1", "ai_provider")
            assert out == "openai-api-key-user-1"

    def test_prod_integration_format(self):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            out = SecretStorageHandler.format_secret_id("jira", "user-2", "integration")
            assert out == "integration-jira-api-key-user-2"


class TestSecretStorageHandlerEncryptDecrypt:
    def test_encrypt_value_roundtrip(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            encrypted = SecretStorageHandler.encrypt_value("secret-value")
            assert encrypted != "secret-value"
            decrypted = SecretStorageHandler.decrypt_value(encrypted)
            assert decrypted == "secret-value"

    def test_decrypt_value_invalid_token_raises(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.decrypt_value("not-valid-fernet-token")
            assert exc_info.value.status_code == 500
            assert "decrypt" in (exc_info.value.detail or "").lower()


class TestSecretStorageHandlerGetSecret:
    def test_get_secret_from_preferences_ai_provider(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            encrypted = SecretStorageHandler.encrypt_value("my-api-key")
            preferences = {"api_key_openai": encrypted}
            with patch.object(
                SecretStorageHandler, "get_client_and_project", return_value=(None, None)
            ):
                result = SecretStorageHandler.get_secret(
                    "openai", "user-1", "ai_provider", db=MagicMock(), preferences=preferences
                )
                assert result == "my-api-key"

    def test_get_secret_from_preferences_integration(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            encrypted = SecretStorageHandler.encrypt_value("jira-key")
            preferences = {"integration_api_key_jira": encrypted}
            with patch.object(
                SecretStorageHandler, "get_client_and_project", return_value=(None, None)
            ):
                result = SecretStorageHandler.get_secret(
                    "jira", "user-1", "integration", db=MagicMock(), preferences=preferences
                )
                assert result == "jira-key"

    def test_get_secret_not_found_raises(self):
        with patch.object(
            SecretStorageHandler, "get_client_and_project", return_value=(None, None)
        ):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.get_secret(
                    "openai", "user-1", "ai_provider", db=MagicMock(), preferences={}
                )
            assert exc_info.value.status_code == 404


class TestSecretStorageHandlerStoreSecret:
    def test_store_secret_fallback_to_preferences(self):
        key = Fernet.generate_key().decode("utf-8")
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": key}, clear=False):
            with patch.object(
                SecretStorageHandler, "get_client_and_project", return_value=(None, None)
            ):
                preferences = {}
                SecretStorageHandler.store_secret(
                    "openai", "user-1", "sk-xyz", "ai_provider", db=MagicMock(), preferences=preferences
                )
                assert "api_key_openai" in preferences
                decrypted = SecretStorageHandler.decrypt_value(preferences["api_key_openai"])
                assert decrypted == "sk-xyz"

    def test_store_secret_no_gcp_no_db_raises(self):
        with patch.object(
            SecretStorageHandler, "get_client_and_project", return_value=(None, None)
        ):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.store_secret(
                    "openai", "user-1", "sk-xyz", "ai_provider", db=None, preferences=None
                )
            assert exc_info.value.status_code == 500
            assert "storage" in (exc_info.value.detail or "").lower() or "available" in (exc_info.value.detail or "").lower()


class TestSecretStorageHandlerCheckSecretExists:
    @pytest.mark.asyncio
    async def test_check_secret_exists_no_gcp_no_db_returns_false(self):
        with patch.object(
            SecretStorageHandler, "get_client_and_project", return_value=(None, None)
        ):
            result = await SecretStorageHandler.check_secret_exists(
                "openai", "user-1", "ai_provider", db=None
            )
            assert result is False


class TestSecretStorageHandlerDeleteSecret:
    def test_delete_secret_no_gcp_no_db_raises_404(self):
        with patch.object(
            SecretStorageHandler, "get_client_and_project", return_value=(None, None)
        ):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.delete_secret("openai", "user-1", "ai_provider", db=None)
            assert exc_info.value.status_code == 404
            assert "No secret" in (exc_info.value.detail or "")
