import pytest
import sys
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

# Mock google.cloud before importing APIKeyService
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.secretmanager"] = MagicMock()

from app.modules.auth.api_key_service import APIKeyService
from app.modules.users.user_model import User
from app.modules.users.user_preferences_model import UserPreferences


class TestAPIKeyService:
    """Test cases for APIKeyService class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def valid_api_key(self):
        """Generate a valid API key for testing."""
        return APIKeyService.generate_api_key()

    @pytest.fixture
    def mock_user_preferences(self):
        """Mock user preferences object."""
        pref = MagicMock(spec=UserPreferences)
        pref.user_id = "test_user_123"
        pref.preferences = {"api_key_hash": "test_hash"}
        return pref

    class TestGenerateAPIKey:
        """Test cases for API key generation."""

        def test_generate_api_key_format(self):
            """Test that generated API key has correct format and prefix."""
            api_key = APIKeyService.generate_api_key()
            assert api_key.startswith(APIKeyService.SECRET_PREFIX)
            assert len(api_key) > len(APIKeyService.SECRET_PREFIX)

        def test_generate_api_key_uniqueness(self):
            """Test that generated API keys are unique."""
            key1 = APIKeyService.generate_api_key()
            key2 = APIKeyService.generate_api_key()
            assert key1 != key2

        def test_generate_api_key_length(self):
            """Test that generated API key has expected length."""
            api_key = APIKeyService.generate_api_key()
            # PREFIX (3 chars) + hex(32 bytes) = 3 + 64 = 67 chars
            expected_length = len(APIKeyService.SECRET_PREFIX) + (
                APIKeyService.KEY_LENGTH * 2
            )
            assert len(api_key) == expected_length

    class TestHashAPIKey:
        """Test cases for API key hashing."""

        def test_hash_api_key_consistent(self):
            """Test that hashing the same key produces the same hash."""
            api_key = "sk-test123456"
            hash1 = APIKeyService.hash_api_key(api_key)
            hash2 = APIKeyService.hash_api_key(api_key)
            assert hash1 == hash2

        def test_hash_api_key_different_keys(self):
            """Test that different keys produce different hashes."""
            key1 = "sk-test123456"
            key2 = "sk-test789012"
            hash1 = APIKeyService.hash_api_key(key1)
            hash2 = APIKeyService.hash_api_key(key2)
            assert hash1 != hash2

        def test_hash_api_key_format(self):
            """Test that hash is in hexadecimal format."""
            api_key = "sk-test123456"
            hashed = APIKeyService.hash_api_key(api_key)
            # SHA256 produces 64 character hex string
            assert len(hashed) == 64
            assert all(c in "0123456789abcdef" for c in hashed)

    class TestValidateAPIKey:
        """Test cases for API key validation."""

        @pytest.mark.asyncio
        async def test_validate_api_key_success(self, mock_db, valid_api_key):
            """Test successful API key validation."""
            # Setup mock database response
            mock_user_pref = MagicMock(spec=UserPreferences)
            mock_user_pref.user_id = "test_user_123"
            mock_email = "test@example.com"

            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = (mock_user_pref, mock_email)

            # Validate API key
            result = await APIKeyService.validate_api_key(valid_api_key, mock_db)

            assert result is not None
            assert result["user_id"] == "test_user_123"
            assert result["email"] == "test@example.com"
            assert result["auth_type"] == "api_key"

        @pytest.mark.asyncio
        async def test_validate_api_key_invalid_prefix(self, mock_db):
            """Test validation fails with invalid prefix."""
            invalid_key = "invalid-key-without-prefix"

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401
            assert "Invalid API key format" in exc_info.value.detail
            assert APIKeyService.SECRET_PREFIX in exc_info.value.detail
            assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}

        @pytest.mark.asyncio
        async def test_validate_api_key_missing_prefix(self, mock_db):
            """Test validation fails when prefix is missing."""
            invalid_key = "just-a-random-string"

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401
            assert "must start with" in exc_info.value.detail

        @pytest.mark.asyncio
        async def test_validate_api_key_empty_string(self, mock_db):
            """Test validation fails with empty string."""
            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key("", mock_db)

            assert exc_info.value.status_code == 401
            assert "Invalid API key format" in exc_info.value.detail

        @pytest.mark.asyncio
        async def test_validate_api_key_not_found(self, mock_db, valid_api_key):
            """Test validation fails when API key not found in database."""
            # Setup mock database to return None (no matching key)
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(valid_api_key, mock_db)

            assert exc_info.value.status_code == 401
            assert "No matching API key found" in exc_info.value.detail
            assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}

        @pytest.mark.asyncio
        async def test_validate_api_key_partial_prefix(self, mock_db):
            """Test validation fails with partial prefix."""
            invalid_key = "sk"  # Just the prefix without rest

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            # Either fails format check or not found - both are acceptable
            assert exc_info.value.status_code == 401

        @pytest.mark.asyncio
        async def test_validate_api_key_wrong_prefix(self, mock_db):
            """Test validation fails with wrong prefix."""
            invalid_key = "pk-1234567890abcdef"  # Wrong prefix

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401
            assert "Invalid API key format" in exc_info.value.detail

        @pytest.mark.asyncio
        async def test_validate_api_key_database_error(self, mock_db, valid_api_key):
            """Test validation handles database errors gracefully."""
            # Setup mock database to raise an exception
            mock_db.query.side_effect = Exception("Database connection error")

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(valid_api_key, mock_db)

            assert exc_info.value.status_code == 500
            assert "Internal error during API key validation" in exc_info.value.detail

        @pytest.mark.asyncio
        async def test_validate_api_key_with_special_characters(self, mock_db):
            """Test validation with special characters in key."""
            invalid_key = "sk-test@#$%^&*()"

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401

        @pytest.mark.asyncio
        async def test_validate_api_key_very_long_key(self, mock_db):
            """Test validation with unusually long key."""
            invalid_key = "sk-" + ("a" * 1000)

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401
            assert "No matching API key found" in exc_info.value.detail

        @pytest.mark.asyncio
        async def test_validate_api_key_whitespace(self, mock_db):
            """Test validation fails with whitespace in key."""
            invalid_key = "sk- with spaces"

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401

        @pytest.mark.asyncio
        async def test_validate_api_key_unicode_characters(self, mock_db):
            """Test validation with unicode characters."""
            invalid_key = "sk-test🔑key"

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401

        @pytest.mark.asyncio
        async def test_validate_api_key_sql_injection_attempt(self, mock_db):
            """Test validation handles SQL injection attempts safely."""
            # Try an SQL injection pattern
            invalid_key = "sk-'; DROP TABLE users; --"

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            # Should fail gracefully without executing malicious SQL
            assert exc_info.value.status_code == 401

        @pytest.mark.asyncio
        async def test_validate_api_key_null_bytes(self, mock_db):
            """Test validation handles null bytes."""
            invalid_key = "sk-test\x00key"

            # Setup mock to return no results
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.params.return_value = mock_query
            mock_query.first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.validate_api_key(invalid_key, mock_db)

            assert exc_info.value.status_code == 401

    class TestCreateAPIKey:
        """Test cases for API key creation."""

        @pytest.mark.asyncio
        async def test_create_api_key_new_user(self, mock_db):
            """Test creating API key for user without existing preferences."""
            user_id = "test_user_123"

            # Mock query to return no existing preferences
            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = None

            # Mock UserPreferences constructor
            mock_pref_instance = MagicMock()
            mock_pref_instance.preferences = {}

            # Mock development mode
            with patch.dict("os.environ", {"isDevelopmentMode": "enabled"}), patch(
                "app.modules.auth.api_key_service.UserPreferences",
                return_value=mock_pref_instance,
            ):
                api_key = await APIKeyService.create_api_key(user_id, mock_db)

                assert api_key.startswith(APIKeyService.SECRET_PREFIX)
                # Verify new UserPreferences was added to session
                mock_db.add.assert_called_once()
                mock_db.commit.assert_called_once()

        @pytest.mark.asyncio
        async def test_create_api_key_existing_user(self, mock_db):
            """Test creating API key for user with existing preferences."""
            user_id = "test_user_123"

            # Mock existing user preferences
            mock_pref = MagicMock(spec=UserPreferences)
            mock_pref.user_id = user_id
            mock_pref.preferences = {}

            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_pref

            # Mock development mode
            with patch.dict("os.environ", {"isDevelopmentMode": "enabled"}):
                api_key = await APIKeyService.create_api_key(user_id, mock_db)

                assert api_key.startswith(APIKeyService.SECRET_PREFIX)
                assert "api_key_hash" in mock_pref.preferences
                mock_db.commit.assert_called_once()

    class TestRevokeAPIKey:
        """Test cases for API key revocation."""

        @pytest.mark.asyncio
        async def test_revoke_api_key_success(self, mock_db):
            """Test successful API key revocation."""
            user_id = "test_user_123"

            # Mock user preferences with API key
            mock_pref = MagicMock(spec=UserPreferences)
            mock_pref.preferences = {"api_key_hash": "test_hash", "other_pref": "value"}

            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_pref

            # Mock development mode
            with patch.dict("os.environ", {"isDevelopmentMode": "enabled"}):
                result = await APIKeyService.revoke_api_key(user_id, mock_db)

                assert result is True
                assert "api_key_hash" not in mock_pref.preferences
                assert "other_pref" in mock_pref.preferences
                mock_db.commit.assert_called_once()

        @pytest.mark.asyncio
        async def test_revoke_api_key_no_user(self, mock_db):
            """Test revoking API key for non-existent user."""
            user_id = "nonexistent_user"

            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = None

            result = await APIKeyService.revoke_api_key(user_id, mock_db)
            assert result is False

        @pytest.mark.asyncio
        async def test_revoke_api_key_no_existing_key(self, mock_db):
            """Test revoking when user has no API key."""
            user_id = "test_user_123"

            # Mock user preferences without API key
            mock_pref = MagicMock(spec=UserPreferences)
            mock_pref.preferences = {"other_pref": "value"}

            mock_query = MagicMock()
            mock_db.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_pref

            result = await APIKeyService.revoke_api_key(user_id, mock_db)
            assert result is True
