import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from sqlalchemy.orm import Session
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.provider.provider_schema import SetProviderRequest
from app.modules.intelligence.provider.llm_config import CHAT_MODEL_CONFIG_MAP, INFERENCE_MODEL_CONFIG_MAP


class TestProviderDualConfig:
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session with user preferences."""
        db = MagicMock()
        preferences = MagicMock()
        preferences.preferences = {
            "selected_chat_model": "GPT-4o Chat",
            "selected_inference_model": "Claude-3 Inference",
            "chat_provider": "openai",
            "inference_provider": "anthropic",
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
    async def test_chat_and_inference_separate_configs(self, provider_service, mock_db):
        """Test that chat and inference configs are separate."""
        # Test chat config
        chat_config = await provider_service.get_global_ai_provider("test_user_id", "chat")
        assert chat_config.config_type == "chat"
        assert chat_config.preferred_llm == "openai"
        assert chat_config.selected_model == "GPT-4o Chat"
        
        # Test inference config
        inference_config = await provider_service.get_global_ai_provider("test_user_id", "inference")
        assert inference_config.config_type == "inference"
        assert inference_config.preferred_llm == "anthropic"
        assert inference_config.selected_model == "Claude-3 Inference"

    @pytest.mark.asyncio
    async def test_dual_provider_config(self, provider_service, mock_db):
        """Test that the dual provider config returns both configs."""
        dual_config = await provider_service.get_dual_provider_config("test_user_id")
        
        # Chat config
        assert dual_config.chat_config.config_type == "chat"
        assert dual_config.chat_config.preferred_llm == "openai"
        assert dual_config.chat_config.selected_model == "GPT-4o Chat"
        
        # Inference config
        assert dual_config.inference_config.config_type == "inference"
        assert dual_config.inference_config.preferred_llm == "anthropic"
        assert dual_config.inference_config.selected_model == "Claude-3 Inference"

    @patch('app.modules.intelligence.provider.provider_service.litellm.acompletion')
    @pytest.mark.asyncio
    async def test_call_llm_uses_correct_config(self, mock_acompletion, provider_service):
        """Test that call_llm uses the correct config based on config_type."""
        mock_acompletion.return_value = AsyncMock()
        mock_acompletion.return_value.choices = [MagicMock()]
        mock_acompletion.return_value.choices[0].message.content = "Test response"
        
        # Call with chat config (default)
        await provider_service.call_llm([{"role": "user", "content": "Hello"}], "small")
        # Check that it used the chat config
        call_args = mock_acompletion.call_args[1]
        assert "gpt-4o-mini" in call_args["model"], f"Should use OpenAI model, got {call_args['model']}"
        
        # Call with inference config
        mock_acompletion.reset_mock()
        await provider_service.call_llm([{"role": "user", "content": "Hello"}], "small", config_type="inference")
        # Check that it used the inference config
        call_args = mock_acompletion.call_args[1]
        assert "claude" in call_args["model"].lower(), f"Should use Claude model, got {call_args['model']}"
        
    @pytest.mark.asyncio
    async def test_set_provider_updates_correct_config(self, provider_service, mock_db):
        """Test that setting a provider updates the correct config."""
        # Mock the necessary methods to avoid actual DB operations
        provider_service.db = mock_db
        mock_db.commit = MagicMock()
        
        # Create a request to set the chat provider
        chat_request = SetProviderRequest(
            provider="gemini",
            selected_model="Gemini Chat",
            config_type="chat"
        )
        
        # Update the chat provider
        await provider_service.set_global_ai_provider("test_user_id", 
                                                      chat_request.provider, 
                                                      None, None, 
                                                      chat_request.config_type, 
                                                      chat_request.selected_model)
        
        # Verify the chat config was updated
        assert provider_service.chat_config.provider == "gemini"
        
        # Create a request to set the inference provider
        inference_request = SetProviderRequest(
            provider="deepseek",
            selected_model="DeepSeek Inference",
            config_type="inference"
        )
        
        # Update the inference provider
        await provider_service.set_global_ai_provider("test_user_id", 
                                                      inference_request.provider, 
                                                      None, None, 
                                                      inference_request.config_type, 
                                                      inference_request.selected_model)
        
        # Verify the inference config was updated
        assert provider_service.inference_config.provider == "deepseek" 