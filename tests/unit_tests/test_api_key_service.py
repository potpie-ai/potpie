import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.auth.api_key_service import APIKeyService
from app.modules.users.user_preferences_model import UserPreferences


@pytest.fixture
def mock_env_vars():
    with patch.dict(
        os.environ, {"GCP_PROJECT": "test-project", "isDevelopmentMode": "disabled"}
    ):
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


class TestAPIKeyService:
    def test_generate_api_key_format(self):
        api_key = APIKeyService.generate_api_key()
        assert api_key.startswith("sk-")
        assert len(api_key) == len("sk-") + (
            APIKeyService.KEY_LENGTH * 2
        )  # *2 because of hex encoding

    def test_hash_api_key(self):
        api_key = "sk-test123"
        hashed_key = APIKeyService.hash_api_key(api_key)
        assert isinstance(hashed_key, str)
        assert len(hashed_key) == 64  # SHA-256 produces 64 character hex string

    def test_get_client_and_project_success(self, mock_env_vars):
        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            client, project_id = APIKeyService.get_client_and_project()
            assert client is not None
            assert project_id == "test-project"
            mock_client.assert_called_once()

    def test_get_client_and_project_dev_mode(self):
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            client, project_id = APIKeyService.get_client_and_project()
            assert client is None
            assert project_id is None

    def test_get_client_and_project_no_project(self):
        with patch.dict(
            os.environ, {"isDevelopmentMode": "disabled", "GCP_PROJECT": ""}
        ):
            with pytest.raises(HTTPException) as exc_info:
                APIKeyService.get_client_and_project()
            assert "GCP_PROJECT environment variable is not set" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_api_key_dev_mode(self, mock_db, mock_user_preferences):
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_user_preferences
            )

            api_key = await APIKeyService.create_api_key("user123", mock_db)

            assert api_key.startswith("sk-")
            assert "api_key_hash" in mock_user_preferences.preferences
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once_with(mock_user_preferences)

    @pytest.mark.asyncio
    async def test_create_api_key_prod_mode(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )

        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            api_key = await APIKeyService.create_api_key("user123", mock_db)

            assert api_key.startswith("sk-")
            assert "api_key_hash" in mock_user_preferences.preferences
            mock_client_instance.create_secret.assert_called_once()
            mock_client_instance.add_secret_version.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once_with(mock_user_preferences)

    @pytest.mark.asyncio
    async def test_create_api_key_new_user(self, mock_env_vars, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_user_preferences = MagicMock(spec=UserPreferences)
        mock_user_preferences.preferences = {}

        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            api_key = await APIKeyService.create_api_key("user123", mock_db)

            assert api_key.startswith("sk-")
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_api_key_gcp_error(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )

        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client_instance.create_secret.side_effect = Exception("GCP Error")
            mock_client.return_value = mock_client_instance

            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.create_api_key("user123", mock_db)

            assert "Failed to store API key" in str(exc_info.value)
            assert "api_key_hash" not in mock_user_preferences.preferences
            mock_db.commit.assert_called()  # Called during rollback

    @pytest.mark.asyncio
    async def test_get_api_key_success(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )
        mock_user_preferences.preferences = {"api_key_hash": "test_hash"}

        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.payload.data = b"sk-test-api-key"
            mock_client_instance.access_secret_version.return_value = mock_response
            mock_client.return_value = mock_client_instance

            api_key = await APIKeyService.get_api_key("user123", mock_db)
            assert api_key == "sk-test-api-key"

    @pytest.mark.asyncio
    async def test_get_api_key_not_found(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )
        mock_user_preferences.preferences = {}

        with patch.dict(
            os.environ, {"isDevelopmentMode": "enabled", "GCP_PROJECT": ""}
        ):
            result = await APIKeyService.get_api_key("user123", mock_db)
            assert result is None

    @pytest.mark.asyncio
    async def test_get_api_key_not_found_prod(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )
        mock_user_preferences.preferences = {"api_key_hash": "some_hash"}

        with patch.dict(
            os.environ, {"isDevelopmentMode": "disabled", "GCP_PROJECT": "test-project"}
        ):
            with pytest.raises(HTTPException) as exc_info:
                await APIKeyService.get_api_key("user123", mock_db)
            assert "Failed to retrieve API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_preferences
        )
        mock_user_preferences.preferences = {"api_key_hash": "test_hash"}

        with patch(
            "google.cloud.secretmanager.SecretManagerServiceClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            success = await APIKeyService.revoke_api_key("user123", mock_db)

            assert success is True
            assert "api_key_hash" not in mock_user_preferences.preferences
            mock_client_instance.delete_secret.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_api_key_not_found(
        self, mock_env_vars, mock_db, mock_user_preferences
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch.dict(
            os.environ, {"isDevelopmentMode": "enabled", "GCP_PROJECT": ""}
        ):
            success = await APIKeyService.revoke_api_key("user123", mock_db)
            assert success is False
