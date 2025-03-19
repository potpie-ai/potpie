import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from sqlalchemy.orm import Session
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.provider.provider_schema import SetProviderRequest
from app.modules.intelligence.provider.llm_config import MODEL_CONFIG_MAP


class TestProviderDualConfig:
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session with user preferences."""
        db = MagicMock()
        preferences = MagicMock()
        preferences.preferences = {
            "chat_model": "openai/gpt-4o",
            "inference_model": "anthropic/claude-3-5-haiku-20241022",
        }
        # Configure db.query().filter().first() to return our mock preferences
        db.query.return_value.filter.return_value.filter_by.return_value.first.return_value = preferences
        return db

    @pytest.fixture
    def provider_service(self, mock_db):
        """Create a provider service with a mock db."""
        with patch('app.modules.intelligence.provider.provider_service.SecretManager'):
            service = ProviderService(mock_db, "test_user_id")
            return service

    @pytest.mark.asyncio
    async def test_get_global_ai_provider(self, provider_service, mock_db):
        """Test getting the global AI provider configuration."""
        config = await provider_service.get_global_ai_provider("test_user_id")
        
        # Verify the configuration
        assert config.chat_model == "openai/gpt-4o"
        assert config.inference_model == "anthropic/claude-3-5-haiku-20241022"
        assert config.provider == "openai"  # Provider should be derived from chat model

    @pytest.mark.asyncio
    async def test_set_global_ai_provider(self, provider_service, mock_db):
        """Test setting the global AI provider configuration."""
        # Create a request to update both models
        request = SetProviderRequest(
            chat_model="anthropic/claude-3-7-sonnet-20250219",
            inference_model="openai/gpt-4o-mini"
        )
        
        # Update the configuration
        await provider_service.set_global_ai_provider("test_user_id", request)
        
        # Verify the chat config was updated
        assert provider_service.chat_config.provider == "anthropic"
        assert provider_service.chat_config.model == "anthropic/claude-3-7-sonnet-20250219"
        
        # Verify the inference config was updated
        assert provider_service.inference_config.provider == "openai"
        assert provider_service.inference_config.model == "openai/gpt-4o-mini"

    @patch('app.modules.intelligence.provider.provider_service.litellm.acompletion')
    @pytest.mark.asyncio
    async def test_call_llm_uses_correct_config(self, mock_acompletion, provider_service):
        """Test that call_llm uses the correct config based on config_type."""
        mock_acompletion.return_value = AsyncMock()
        mock_acompletion.return_value.choices = [MagicMock()]
        mock_acompletion.return_value.choices[0].message.content = "Test response"
        
        # Call with chat config (default)
        await provider_service.call_llm([{"role": "user", "content": "Hello"}])
        # Check that it used the chat config
        call_args = mock_acompletion.call_args[1]
        assert "openai/gpt-4o" == call_args["model"], f"Should use OpenAI model, got {call_args['model']}"
        
        # Call with inference config
        mock_acompletion.reset_mock()
        await provider_service.call_llm([{"role": "user", "content": "Hello"}], config_type="inference")
        # Check that it used the inference config
        call_args = mock_acompletion.call_args[1]
        assert "anthropic/claude-3-5-haiku-20241022" == call_args["model"], f"Should use Claude model, got {call_args['model']}" 