import unittest
from unittest import mock
import asyncio # For async stream testing

# Actual imports for ProviderService and its dependencies
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.provider.llm_config import LLMProviderConfig, LLMProvider # Assuming these are used for config mock

# Mock relevant dependencies, e.g., db session
from sqlalchemy.orm import Session # For type hinting Session

# Need to ensure app.core.telemetry.get_tracer is available for patching
import app.core.telemetry 
import litellm # To patch litellm.acompletion

class TestProviderServiceTelemetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_tracer = mock.MagicMock()
        self.mock_span = mock.MagicMock()
        # Configure the mock span to be a context manager
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span
        self.mock_tracer.start_as_current_span.return_value.__exit__ = mock.MagicMock(return_value=None)


        self.get_tracer_patcher = mock.patch(
            'app.modules.intelligence.provider.provider_service.get_tracer', 
            return_value=self.mock_tracer
        )
        self.mock_get_tracer = self.get_tracer_patcher.start()

        self.mock_db = mock.MagicMock(spec=Session)
        self.user_id = "test_provider_user"

        # Mock the config objects chat_config and inference_config
        # These should mimic LLMProviderConfig structure
        # Note: model_name in LLMProviderConfig is the full path like "openai/gpt-test-chat"
        self.mock_chat_config = LLMProviderConfig(
            provider=LLMProvider.OPENAI, model_name="openai/gpt-test-chat", api_key="fake_key"
        )
        self.mock_inference_config = LLMProviderConfig(
            provider=LLMProvider.ANTHROPIC, model_name="anthropic/claude-test-inf", api_key="fake_key"
        )
        
        # Let's mock the LiteLLM acompletion call that is used internally by the service
        self.mock_litellm_acompletion = mock.AsyncMock() # Changed to just mock_litellm_acompletion
        self.litellm_patcher = mock.patch('litellm.acompletion', new=self.mock_litellm_acompletion)
        self.litellm_patcher.start() # Start the patcher

        self.provider_service = ProviderService(db=self.mock_db, user_id=self.user_id)
        # After ProviderService is initialized, set its config attributes
        self.provider_service.chat_config = self.mock_chat_config
        self.provider_service.inference_config = self.mock_inference_config

        # Mock _build_llm_params to return something simple, as it involves API key lookups etc.
        # It's called by call_llm before acompletion.
        # It needs to return a dict that includes the 'model' key for litellm.
        self.build_llm_params_patcher = mock.patch(
            'app.modules.intelligence.provider.provider_service.ProviderService._build_llm_params',
            # Return a dict that would be updated by extra_params. The key is that it must have 'model'.
            return_value={'model': 'mock_model_from_build_params', 'temperature': 0.1} 
        )
        self.mock_build_llm_params = self.build_llm_params_patcher.start()
        
        # Also mock get_extra_params_and_headers as it's called by call_llm
        self.get_extra_params_patcher = mock.patch(
            'app.modules.intelligence.provider.provider_service.ProviderService.get_extra_params_and_headers',
            return_value=({}, None) # Return empty dict for extra_params and None for headers
        )
        self.mock_get_extra_params = self.get_extra_params_patcher.start()


    def tearDown(self):
        self.get_tracer_patcher.stop()
        self.litellm_patcher.stop()
        self.build_llm_params_patcher.stop()
        self.get_extra_params_patcher.stop()

    async def test_call_llm_no_stream_success_creates_span(self):
        messages = [{"role": "user", "content": "Hello"}]
        config_type = "chat"
        
        mock_response_object = mock.MagicMock()
        mock_response_object.choices = [mock.MagicMock(message=mock.MagicMock(content="Hi there"))]
        mock_response_object.usage = mock.MagicMock(prompt_tokens=10, completion_tokens=5)
        self.mock_litellm_acompletion.return_value = mock_response_object

        # Ensure _build_llm_params returns a dict with the correct model for this config_type
        self.mock_build_llm_params.return_value = {'model': self.mock_chat_config.model_name, 'temperature': 0.1}

        response_text = await self.provider_service.call_llm(messages=messages, stream=False, config_type=config_type)

        self.assertEqual(response_text, "Hi there")
        self.mock_tracer.start_as_current_span.assert_called_once_with("llm.call")
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}
        
        self.assertEqual(actual_attributes.get("user.id"), self.user_id)
        self.assertEqual(actual_attributes.get("llm.config_type"), config_type)
        self.assertEqual(actual_attributes.get("llm.model_name"), self.mock_chat_config.model_name)
        self.assertEqual(actual_attributes.get("llm.provider"), self.mock_chat_config.provider.value)
        self.assertEqual(actual_attributes.get("llm.stream"), False)
        self.assertEqual(actual_attributes.get("token.usage.input"), 10)
        self.assertEqual(actual_attributes.get("token.usage.output"), 5)
        self.assertEqual(actual_attributes.get("llm.status"), "success")
        
        self.mock_span.record_exception.assert_not_called()
        self.mock_litellm_acompletion.assert_called_once()

    async def test_call_llm_failure_records_exception(self):
        messages = [{"role": "user", "content": "Hello"}]
        config_type = "chat"
        test_exception = Exception("LLM API Error")
        self.mock_litellm_acompletion.side_effect = test_exception
        
        self.mock_build_llm_params.return_value = {'model': self.mock_chat_config.model_name, 'temperature': 0.1}

        with self.assertRaises(Exception) as context:
            await self.provider_service.call_llm(messages=messages, stream=False, config_type=config_type)
        
        self.assertEqual(context.exception, test_exception)

        self.mock_tracer.start_as_current_span.assert_called_once_with("llm.call")
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}

        self.assertEqual(actual_attributes.get("llm.status"), "failure")
        self.assertEqual(actual_attributes.get("llm.model_name"), self.mock_chat_config.model_name)
        self.assertEqual(actual_attributes.get("user.id"), self.user_id)
        self.assertEqual(actual_attributes.get("llm.config_type"), config_type)
        
        self.mock_span.record_exception.assert_called_once_with(test_exception)

    async def test_call_llm_stream_success_updates_span_status(self):
        messages = [{"role": "user", "content": "Stream me"}]
        config_type = "inference"

        async def mock_stream_generator():
            yield mock.MagicMock(choices=[mock.MagicMock(delta=mock.MagicMock(content="chunk1"))])
            yield mock.MagicMock(choices=[mock.MagicMock(delta=mock.MagicMock(content="chunk2"))])
        
        self.mock_litellm_acompletion.return_value = mock_stream_generator()
        self.mock_build_llm_params.return_value = {'model': self.mock_inference_config.model_name, 'temperature': 0.1}

        response_stream = await self.provider_service.call_llm(messages=messages, stream=True, config_type=config_type)
        
        results = []
        async for chunk in response_stream:
            results.append(chunk)

        self.assertEqual(results, ["chunk1", "chunk2"])
        self.mock_tracer.start_as_current_span.assert_called_once_with("llm.call")
        
        status_attributes = [call[0] for call in self.mock_span.set_attribute.call_args_list if call[0][0] == "llm.status"]
        self.assertIn(("llm.status", "streaming_started"), status_attributes)
        self.assertIn(("llm.status", "success"), status_attributes)

        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}
        self.assertEqual(actual_attributes.get("llm.model_name"), self.mock_inference_config.model_name)
        self.assertEqual(actual_attributes.get("llm.provider"), self.mock_inference_config.provider.value)
        self.assertEqual(actual_attributes.get("llm.stream"), True)
        self.mock_span.record_exception.assert_not_called()

    async def test_call_llm_stream_failure_updates_status_and_records_exception(self):
        messages = [{"role": "user", "content": "Stream me error"}]
        config_type = "chat"
        test_exception = ValueError("Stream error")

        async def mock_error_stream_generator():
            yield mock.MagicMock(choices=[mock.MagicMock(delta=mock.MagicMock(content="good_chunk"))])
            raise test_exception
        
        self.mock_litellm_acompletion.return_value = mock_error_stream_generator()
        self.mock_build_llm_params.return_value = {'model': self.mock_chat_config.model_name, 'temperature': 0.1}

        response_stream = await self.provider_service.call_llm(messages=messages, stream=True, config_type=config_type)

        with self.assertRaises(ValueError) as context:
            async for _ in response_stream:
                pass # Consume the stream to trigger the exception
        self.assertEqual(context.exception, test_exception)

        self.mock_tracer.start_as_current_span.assert_called_once_with("llm.call")
        status_attributes = [call[0] for call in self.mock_span.set_attribute.call_args_list if call[0][0] == "llm.status"]
        self.assertIn(("llm.status", "streaming_started"), status_attributes)
        self.assertIn(("llm.status", "failure_during_stream"), status_attributes)
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}
        self.assertEqual(actual_attributes.get("llm.model_name"), self.mock_chat_config.model_name)
        self.assertEqual(actual_attributes.get("llm.provider"), self.mock_chat_config.provider.value)
        self.assertEqual(actual_attributes.get("llm.stream"), True)
        
        self.mock_span.record_exception.assert_called_once_with(test_exception)

if __name__ == '__main__':
    unittest.main()
