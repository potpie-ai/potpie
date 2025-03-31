import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.key_management.secrets_schema import BaseSecret

# Mock responses
MOCK_SECRET_KEY = Fernet.generate_key()
MOCK_API_KEY = "test-api-key-123"
MOCK_ENCRYPTED_KEY = Fernet(MOCK_SECRET_KEY).encrypt(MOCK_API_KEY.encode()).decode()

@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        'SECRET_ENCRYPTION_KEY': MOCK_SECRET_KEY.decode(),
        'GCP_PROJECT': 'test-project'
    }):
        yield

@pytest.fixture
def mock_db():
    db = MagicMock(spec=Session)
    return db

@pytest.fixture
def mock_user_preferences():
    pref = MagicMock(spec=UserPreferences)
    pref.preferences = {}
    return pref

class TestSecretManager:
    def test_get_fernet_success(self, mock_env_vars):
        fernet = SecretManager._get_fernet()
        assert isinstance(fernet, Fernet)

    def test_get_fernet_missing_key(self):
        with patch.dict(os.environ, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                SecretManager._get_fernet()
            assert "SECRET_ENCRYPTION_KEY environment variable is not set" in str(exc_info.value)

    def test_get_fernet_invalid_key(self):
        with patch.dict(os.environ, {'SECRET_ENCRYPTION_KEY': 'invalid-key'}):
            with pytest.raises(HTTPException) as exc_info:
                SecretManager._get_fernet()
            assert "Invalid SECRET_ENCRYPTION_KEY" in str(exc_info.value)

    def test_encrypt_decrypt_api_key_success(self, mock_env_vars):
        encrypted = SecretManager.encrypt_api_key(MOCK_API_KEY)
        decrypted = SecretManager.decrypt_api_key(encrypted)
        assert decrypted == MOCK_API_KEY

    def test_decrypt_api_key_invalid_token(self, mock_env_vars):
        with pytest.raises(HTTPException) as exc_info:
            SecretManager.decrypt_api_key("invalid-encrypted-key")
        assert "Failed to decrypt API key" in str(exc_info.value)

    def test_get_client_and_project_success(self, mock_env_vars):
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            client, project_id = SecretManager.get_client_and_project()
            assert client is not None
            assert project_id == 'test-project'
            mock_client.assert_called_once()

    def test_get_client_and_project_no_project(self):
        with patch.dict(os.environ, clear=True):
            client, project_id = SecretManager.get_client_and_project()
            assert client is None
            assert project_id is None

    def test_get_secret_id_development_mode(self):
        with patch.dict(os.environ, {'isDevelopmentMode': 'enabled'}):
            secret_id = SecretManager.get_secret_id('openai', 'user123')
            assert secret_id is None

    def test_get_secret_id_production_mode(self):
        with patch.dict(os.environ, {'isDevelopmentMode': 'disabled'}):
            secret_id = SecretManager.get_secret_id('openai', 'user123')
            assert secret_id == 'openai-api-key-user123'

    @pytest.mark.asyncio
    async def test_check_secret_exists_for_user_gcp_success(self, mock_env_vars, mock_db):
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            exists = await SecretManager.check_secret_exists_for_user('openai', 'user123', mock_db)
            assert exists is True

    @pytest.mark.asyncio
    async def test_check_secret_exists_for_user_db_fallback(self, mock_db, mock_user_preferences):
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_preferences
        mock_user_preferences.preferences = {'api_key_openai': MOCK_ENCRYPTED_KEY}
        
        exists = await SecretManager.check_secret_exists_for_user('openai', 'user123', mock_db)
        assert exists is True

    def test_process_config_gcp_success(self, mock_env_vars):
        config = BaseSecret(api_key=MOCK_API_KEY, model="openai/gpt-4")
        preferences = {}
        updated_providers = []
        
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            SecretManager._process_config(
                config=config,
                config_type="chat",
                customer_id="user123",
                client=mock_client_instance,
                project_id="test-project",
                preferences=preferences,
                updated_providers=updated_providers
            )
            
            assert preferences["chat_model"] == "openai/gpt-4"
            assert "openai" in updated_providers
            mock_client_instance.add_secret_version.assert_called_once()

    def test_process_config_db_fallback(self, mock_env_vars):
        config = BaseSecret(api_key=MOCK_API_KEY, model="openai/gpt-4")
        preferences = {}
        updated_providers = []
        
        SecretManager._process_config(
            config=config,
            config_type="chat",
            customer_id="user123",
            client=None,
            project_id=None,
            preferences=preferences,
            updated_providers=updated_providers
        )
        
        assert preferences["chat_model"] == "openai/gpt-4"
        assert "openai" in updated_providers
        assert preferences["api_key_openai"] is not None

    def test_get_secret_gcp_success(self, mock_env_vars, mock_db):
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.payload.data = MOCK_API_KEY.encode()
            mock_client_instance.access_secret_version.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            result = SecretManager.get_secret('openai', 'user123', mock_db)
            
            assert result["api_key"] == MOCK_API_KEY
            assert result["provider"] == "openai"

    def test_get_secret_db_fallback(self, mock_env_vars, mock_db, mock_user_preferences):
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_preferences
        mock_user_preferences.preferences = {'api_key_openai': MOCK_ENCRYPTED_KEY}
        
        with patch.dict(os.environ, {'GCP_PROJECT': ''}):
            result = SecretManager.get_secret('openai', 'user123', mock_db)
            
            assert result["api_key"] == MOCK_API_KEY
            assert result["provider"] == "openai"

    def test_get_secret_not_found(self, mock_env_vars, mock_db, mock_user_preferences):
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user_preferences
        mock_user_preferences.preferences = {}
        
        with patch.dict(os.environ, {'GCP_PROJECT': ''}):
            with pytest.raises(HTTPException) as exc_info:
                SecretManager.get_secret('openai', 'user123', mock_db)
            assert "Secret not found in UserPreferences" in str(exc_info.value) 