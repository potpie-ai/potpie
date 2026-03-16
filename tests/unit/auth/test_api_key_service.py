"""
Unit tests for APIKeyService (app/modules/auth/api_key_service.py)

Tests cover:
- API key generation and hashing
- Encryption/decryption for local storage
- create_api_key with mocked DB and GCP
- validate_api_key
- revoke_api_key
- get_api_key
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from cryptography.fernet import Fernet

from app.modules.auth.api_key_service import APIKeyService


pytestmark = pytest.mark.unit


class TestAPIKeyGeneration:
    """Tests for API key generation and hashing"""

    def test_generate_api_key_format(self):
        """Test generated API key has correct prefix and length"""
        api_key = APIKeyService.generate_api_key()

        assert api_key.startswith("sk-")
        # sk- prefix (3) + 64 hex chars (32 bytes * 2)
        assert len(api_key) == 3 + 64

    def test_generate_api_key_uniqueness(self):
        """Test generated API keys are unique"""
        keys = [APIKeyService.generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_hash_api_key(self):
        """Test API key hashing produces consistent results"""
        api_key = "sk-test1234567890abcdef"

        hash1 = APIKeyService.hash_api_key(api_key)
        hash2 = APIKeyService.hash_api_key(api_key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_hash_api_key_different_keys(self):
        """Test different keys produce different hashes"""
        key1 = "sk-key1"
        key2 = "sk-key2"

        hash1 = APIKeyService.hash_api_key(key1)
        hash2 = APIKeyService.hash_api_key(key2)

        assert hash1 != hash2


class TestGetClientAndProject:
    """Tests for get_client_and_project"""

    def test_get_client_dev_mode(self):
        """Test returns None in development mode"""
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            client, project = APIKeyService.get_client_and_project()

            assert client is None
            assert project is None

    def test_get_client_gcp_disabled(self):
        """Test returns None when GCP is disabled"""
        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "disabled", "GCP_SECRET_MANAGER_DISABLED": "true"},
            clear=False,
        ):
            client, project = APIKeyService.get_client_and_project()

            assert client is None
            assert project is None

    def test_get_client_no_project_id(self):
        """Test returns None when GCP_PROJECT not set"""
        env = {"isDevelopmentMode": "disabled", "GCP_SECRET_MANAGER_DISABLED": "false"}
        # Remove GCP_PROJECT if it exists
        with patch.dict(os.environ, env, clear=False):
            with patch.dict(os.environ, {"GCP_PROJECT": ""}, clear=False):
                client, project = APIKeyService.get_client_and_project()

                assert client is None
                assert project is None


class TestEncryption:
    """Tests for encryption/decryption"""

    @pytest.fixture
    def valid_fernet_key(self):
        """Generate a valid Fernet key for testing"""
        return Fernet.generate_key().decode("utf-8")

    def test_get_encryption_key_missing(self):
        """Test raises error when encryption key not set"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": ""}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                APIKeyService.get_encryption_key()

            assert exc_info.value.status_code == 500
            assert "SECRET_ENCRYPTION_KEY" in exc_info.value.detail

    def test_get_encryption_key_invalid(self):
        """Test raises error for invalid encryption key"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": "not-valid-key"}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                APIKeyService.get_encryption_key()

            assert exc_info.value.status_code == 500
            assert "Invalid SECRET_ENCRYPTION_KEY" in exc_info.value.detail

    def test_encrypt_decrypt_roundtrip(self, valid_fernet_key):
        """Test encrypt and decrypt produce original value"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": valid_fernet_key}, clear=False):
            original = "sk-mysecretapikey123"

            encrypted = APIKeyService.encrypt_value(original)
            decrypted = APIKeyService.decrypt_value(encrypted)

            assert decrypted == original
            assert encrypted != original

    def test_decrypt_invalid_token(self, valid_fernet_key):
        """Test decrypt raises error for invalid token"""
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": valid_fernet_key}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                APIKeyService.decrypt_value("not-valid-encrypted-data")

            assert exc_info.value.status_code == 500
            assert "Invalid token" in exc_info.value.detail


class TestCreateAPIKey:
    """Tests for create_api_key"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = MagicMock()
        return db

    @pytest.fixture
    def mock_user_pref(self):
        """Create a mock user preferences object"""
        pref = MagicMock()
        pref.preferences = {}
        return pref

    @pytest.fixture
    def valid_fernet_key(self):
        return Fernet.generate_key().decode("utf-8")

    @pytest.mark.asyncio
    async def test_create_api_key_local_storage(self, mock_db, mock_user_pref, valid_fernet_key):
        """Test creating API key with local storage fallback"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "enabled", "SECRET_ENCRYPTION_KEY": valid_fernet_key},
            clear=False,
        ):
            api_key = await APIKeyService.create_api_key("user-123", mock_db)

            assert api_key.startswith("sk-")
            assert "api_key_hash" in mock_user_pref.preferences
            mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_create_api_key_new_user_pref(self, mock_db, valid_fernet_key):
        """Test creating API key when user has no preferences"""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch.dict(
            os.environ,
            {"isDevelopmentMode": "enabled", "SECRET_ENCRYPTION_KEY": valid_fernet_key},
            clear=False,
        ):
            api_key = await APIKeyService.create_api_key("new-user", mock_db)

            assert api_key.startswith("sk-")
            mock_db.add.assert_called_once()


class TestValidateAPIKey:
    """Tests for validate_api_key"""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_prefix(self, mock_db):
        """Test validation fails for key without correct prefix"""
        result = await APIKeyService.validate_api_key("invalid-key", mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_not_found(self, mock_db):
        """Test validation fails when key not found"""
        mock_db.query.return_value.join.return_value.filter.return_value.params.return_value.first.return_value = None

        result = await APIKeyService.validate_api_key("sk-validformat", mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, mock_db):
        """Test successful API key validation"""
        mock_user_pref = MagicMock()
        mock_user_pref.user_id = "user-123"
        mock_db.query.return_value.join.return_value.filter.return_value.params.return_value.first.return_value = (
            mock_user_pref,
            "user@example.com",
        )

        result = await APIKeyService.validate_api_key("sk-validkey", mock_db)

        assert result is not None
        assert result["user_id"] == "user-123"
        assert result["email"] == "user@example.com"
        assert result["auth_type"] == "api_key"

    @pytest.mark.asyncio
    async def test_validate_api_key_db_error(self, mock_db):
        """Test validation handles database errors"""
        mock_db.query.side_effect = Exception("DB connection failed")

        result = await APIKeyService.validate_api_key("sk-somekey", mock_db)

        assert result is None


class TestRevokeAPIKey:
    """Tests for revoke_api_key"""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_revoke_api_key_no_user_pref(self, mock_db):
        """Test revoke returns False when user has no preferences"""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = await APIKeyService.revoke_api_key("user-123", mock_db)

        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(self, mock_db):
        """Test successful API key revocation"""
        mock_user_pref = MagicMock()
        mock_user_pref.preferences = {
            "api_key_hash": "somehash",
            "encrypted_api_key": "someencrypted",
            "other_pref": "keep-this",
        }
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
            result = await APIKeyService.revoke_api_key("user-123", mock_db)

            assert result is True
            assert "api_key_hash" not in mock_user_pref.preferences
            assert "encrypted_api_key" not in mock_user_pref.preferences
            assert "other_pref" in mock_user_pref.preferences
            mock_db.commit.assert_called()


class TestGetAPIKey:
    """Tests for get_api_key"""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def valid_fernet_key(self):
        return Fernet.generate_key().decode("utf-8")

    @pytest.mark.asyncio
    async def test_get_api_key_no_user_pref(self, mock_db):
        """Test get_api_key returns None when no preferences"""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = await APIKeyService.get_api_key("user-123", mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_api_key_no_hash(self, mock_db):
        """Test get_api_key returns None when no api_key_hash"""
        mock_user_pref = MagicMock()
        mock_user_pref.preferences = {"other_pref": "value"}
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

        result = await APIKeyService.get_api_key("user-123", mock_db)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_api_key_local_storage(self, mock_db, valid_fernet_key):
        """Test get_api_key retrieves from local encrypted storage"""
        # First encrypt a key
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": valid_fernet_key}, clear=False):
            encrypted = APIKeyService.encrypt_value("sk-myapikey")

            mock_user_pref = MagicMock()
            mock_user_pref.preferences = {
                "api_key_hash": "somehash",
                "encrypted_api_key": encrypted,
            }
            mock_db.query.return_value.filter.return_value.first.return_value = mock_user_pref

            with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}, clear=False):
                result = await APIKeyService.get_api_key("user-123", mock_db)

                assert result == "sk-myapikey"
