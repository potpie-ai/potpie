"""
Unit tests for SecretStorageHandler (app/modules/key_management/secret_manager.py)

Tests cover:
- GCP availability caching
- Encryption/decryption
- store_secret with GCP and fallback
- get_secret with GCP and fallback
- delete_secret
- format_secret_id
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from cryptography.fernet import Fernet

from app.modules.key_management.secret_manager import SecretStorageHandler


pytestmark = pytest.mark.unit


class TestGCPAvailability:
    """Tests for GCP availability checking and caching"""

    def setup_method(self):
        """Reset the GCP availability cache before each test"""
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

    def test_gcp_disabled_via_env(self):
        """Test GCP is disabled when GCP_SECRET_MANAGER_DISABLED is set"""
        with patch.dict(os.environ, {"GCP_SECRET_MANAGER_DISABLED": "true"}, clear=False):
            client, project_id = SecretStorageHandler.get_client_and_project()

            assert client is None
            assert project_id is None
            assert SecretStorageHandler._gcp_available is False

    def test_gcp_no_project_id(self):
        """Test GCP unavailable when GCP_PROJECT not set"""
        with patch.dict(
            os.environ,
            {"GCP_SECRET_MANAGER_DISABLED": "false", "GCP_PROJECT": ""},
            clear=False,
        ):
            client, project_id = SecretStorageHandler.get_client_and_project()

            assert client is None
            assert project_id is None

    def test_gcp_client_creation_fails(self):
        """Test GCP unavailable when client creation fails"""
        with patch.dict(
            os.environ,
            {"GCP_SECRET_MANAGER_DISABLED": "false", "GCP_PROJECT": "test-project"},
            clear=False,
        ):
            with patch(
                "app.modules.key_management.secret_manager.secretmanager.SecretManagerServiceClient",
                side_effect=Exception("No credentials"),
            ):
                client, project_id = SecretStorageHandler.get_client_and_project()

                assert client is None
                assert project_id is None
                assert SecretStorageHandler._gcp_available is False

    def test_gcp_cached_result(self):
        """Test GCP availability result is cached"""
        # Set cached result
        SecretStorageHandler._gcp_available = True
        SecretStorageHandler._gcp_client = MagicMock()
        SecretStorageHandler._gcp_project_id = "cached-project"

        client, project_id = SecretStorageHandler.get_client_and_project()

        assert project_id == "cached-project"
        assert client is not None


class TestEncryption:
    """Tests for encryption/decryption"""

    @pytest.fixture
    def valid_fernet_key(self):
        return Fernet.generate_key().decode("utf-8")

    def test_get_encryption_key_missing(self):
        """Test raises error when SECRET_ENCRYPTION_KEY not set"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": ""}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.get_encryption_key()

            assert exc_info.value.status_code == 500

    def test_encrypt_decrypt_roundtrip(self, valid_fernet_key):
        """Test encrypt and decrypt produce original value"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": valid_fernet_key}, clear=False):
            original = "my-secret-value"

            encrypted = SecretStorageHandler.encrypt_value(original)
            decrypted = SecretStorageHandler.decrypt_value(encrypted)

            assert decrypted == original
            assert encrypted != original

    def test_decrypt_invalid_token(self, valid_fernet_key):
        """Test decrypt raises error for invalid encrypted data"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": valid_fernet_key}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.decrypt_value("invalid-encrypted-data")

            assert exc_info.value.status_code == 500


class TestFormatSecretId:
    """Tests for format_secret_id"""

    def test_format_secret_id_ai_provider(self):
        """Test secret ID format for AI provider"""
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            secret_id = SecretStorageHandler.format_secret_id(
                "openai", "user-123", "ai_provider"
            )

            assert secret_id == "openai-api-key-user-123"

    def test_format_secret_id_integration(self):
        """Test secret ID format for integration"""
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}, clear=False):
            secret_id = SecretStorageHandler.format_secret_id(
                "jira", "user-456", "integration"
            )

            assert secret_id == "integration-jira-api-key-user-456"

    def test_format_secret_id_dev_mode(self):
        """Test format_secret_id returns None in dev mode"""
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            secret_id = SecretStorageHandler.format_secret_id(
                "openai", "user-123", "ai_provider"
            )

            assert secret_id is None


class TestStoreSecret:
    """Tests for store_secret"""

    def setup_method(self):
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

    @pytest.fixture
    def valid_fernet_key(self):
        return Fernet.generate_key().decode("utf-8")

    def test_store_secret_fallback_to_db(self, valid_fernet_key):
        """Test store_secret uses DB fallback when GCP unavailable"""
        # Reset GCP cache to force dev mode behavior
        SecretStorageHandler._gcp_available = False
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

        preferences = {}
        mock_db = MagicMock()

        with patch.dict(
            os.environ,
            {
                "isDevelopmentMode": "enabled",
                "SECRET_ENCRYPTION_KEY": valid_fernet_key,
                "GCP_SECRET_MANAGER_DISABLED": "true",
            },
            clear=False,
        ):
            SecretStorageHandler.store_secret(
                service="openai",
                customer_id="user-123",
                api_key="sk-test-key",
                service_type="ai_provider",
                db=mock_db,
                preferences=preferences,
            )

            assert "api_key_openai" in preferences
            # Verify it's encrypted (different from original)
            assert preferences["api_key_openai"] != "sk-test-key"

    def test_store_secret_integration_type(self, valid_fernet_key):
        """Test store_secret uses correct key name for integration type"""
        # Reset GCP cache
        SecretStorageHandler._gcp_available = False
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

        preferences = {}
        mock_db = MagicMock()

        with patch.dict(
            os.environ,
            {
                "isDevelopmentMode": "enabled",
                "SECRET_ENCRYPTION_KEY": valid_fernet_key,
                "GCP_SECRET_MANAGER_DISABLED": "true",
            },
            clear=False,
        ):
            SecretStorageHandler.store_secret(
                service="jira",
                customer_id="user-123",
                api_key="jira-api-key",
                service_type="integration",
                db=mock_db,
                preferences=preferences,
            )

            assert "integration_api_key_jira" in preferences

    def test_store_secret_no_storage_available(self):
        """Test store_secret raises error when no storage available"""
        # Reset GCP cache
        SecretStorageHandler._gcp_available = False
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "enabled", "GCP_SECRET_MANAGER_DISABLED": "true"},
            clear=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.store_secret(
                    service="openai",
                    customer_id="user-123",
                    api_key="sk-test",
                    db=None,
                    preferences=None,
                )

            assert exc_info.value.status_code == 500
            assert "Neither GCP nor database" in exc_info.value.detail


class TestGetSecret:
    """Tests for get_secret"""

    def setup_method(self):
        SecretStorageHandler._gcp_available = None
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

    @pytest.fixture
    def valid_fernet_key(self):
        return Fernet.generate_key().decode("utf-8")

    def test_get_secret_from_preferences(self, valid_fernet_key):
        """Test get_secret retrieves from preferences"""
        with patch.dict(
            os.environ,
            {
                "isDevelopmentMode": "enabled",
                "SECRET_ENCRYPTION_KEY": valid_fernet_key,
            },
            clear=False,
        ):
            # First encrypt the key
            encrypted = SecretStorageHandler.encrypt_value("sk-original-key")
            preferences = {"api_key_openai": encrypted}
            mock_db = MagicMock()

            result = SecretStorageHandler.get_secret(
                service="openai",
                customer_id="user-123",
                service_type="ai_provider",
                db=mock_db,
                preferences=preferences,
            )

            assert result == "sk-original-key"

    def test_get_secret_integration_type(self, valid_fernet_key):
        """Test get_secret uses correct key name for integration type"""
        with patch.dict(
            os.environ,
            {
                "isDevelopmentMode": "enabled",
                "SECRET_ENCRYPTION_KEY": valid_fernet_key,
            },
            clear=False,
        ):
            encrypted = SecretStorageHandler.encrypt_value("jira-secret")
            preferences = {"integration_api_key_jira": encrypted}
            mock_db = MagicMock()

            result = SecretStorageHandler.get_secret(
                service="jira",
                customer_id="user-123",
                service_type="integration",
                db=mock_db,
                preferences=preferences,
            )

            assert result == "jira-secret"

    def test_get_secret_not_found(self):
        """Test get_secret raises 404 when secret not found"""
        preferences = {}
        mock_db = MagicMock()

        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.get_secret(
                    service="openai",
                    customer_id="user-456",  # Not test-user
                    service_type="ai_provider",
                    db=mock_db,
                    preferences=preferences,
                )

            assert exc_info.value.status_code == 404

    def test_get_secret_test_user_returns_none(self):
        """Test get_secret returns None silently for test-user"""
        preferences = {}
        mock_db = MagicMock()

        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            result = SecretStorageHandler.get_secret(
                service="openai",
                customer_id="test-user",
                service_type="ai_provider",
                db=mock_db,
                preferences=preferences,
            )

            assert result is None


class TestDeleteSecret:
    """Tests for delete_secret"""

    def setup_method(self):
        SecretStorageHandler._gcp_available = False
        SecretStorageHandler._gcp_client = None
        SecretStorageHandler._gcp_project_id = None

    def test_delete_secret_from_db(self):
        """Test delete_secret removes from database"""
        mock_user_pref = MagicMock()
        mock_user_pref.preferences = {
            "api_key_openai": "encrypted-value",
            "other_pref": "keep-this",
        }

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "enabled", "GCP_SECRET_MANAGER_DISABLED": "true"},
            clear=False,
        ):
            result = SecretStorageHandler.delete_secret(
                service="openai",
                customer_id="user-123",
                service_type="ai_provider",
                db=mock_db,
            )

            assert result is True
            assert "api_key_openai" not in mock_user_pref.preferences
            assert "other_pref" in mock_user_pref.preferences
            mock_db.commit.assert_called()

    def test_delete_secret_not_found_raises(self):
        """Test delete_secret raises HTTPException when secret not found"""
        mock_user_pref = MagicMock()
        mock_user_pref.preferences = {}  # No API key

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "enabled", "GCP_SECRET_MANAGER_DISABLED": "true"},
            clear=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.delete_secret(
                    service="nonexistent",
                    customer_id="user-123",
                    service_type="ai_provider",
                    db=mock_db,
                )

            assert exc_info.value.status_code == 404
            assert "No secret found" in exc_info.value.detail
