import unittest
from unittest import mock
import json # For comparing JSON string attributes
import asyncio # For asyncio.create_task if needed for CodeProviderService mocking

# Actual imports for ConversationService and its dependencies
from app.modules.conversations.conversation.conversation_service import ConversationService, ConversationServiceError
from sqlalchemy.exc import IntegrityError # For simulating DB errors
from app.modules.conversations.conversation.conversation_schema import CreateConversationRequest

# Mock relevant dependencies of ConversationService (db session, various services, etc.)
from sqlalchemy.orm import Session # For type hinting Session
from app.modules.projects.projects_service import ProjectService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.provider.provider_service import ProviderService as LLMProviderService # Renamed to avoid conflict
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.agents.custom_agents.custom_agents_service import CustomAgentService
# Assuming PromptService is the correct name, if it's 'promt_service' in code, that will be used in setup
from app.modules.intelligence.agents.chat_agents.adaptive_agent import PromptService

# Need to ensure app.core.telemetry.get_tracer is available for patching
import app.core.telemetry

class TestConversationServiceTelemetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_tracer = mock.MagicMock()
        self.mock_span = mock.MagicMock()
        # Configure the context manager mock for the span
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span
        self.mock_tracer.start_as_current_span.return_value.__exit__ = mock.MagicMock(return_value=None) # Ensure __exit__ is also a mock

        self.get_tracer_patcher = mock.patch(
            'app.modules.conversations.conversation.conversation_service.get_tracer', 
            return_value=self.mock_tracer
        )
        self.mock_get_tracer = self.get_tracer_patcher.start()

        # Mock all dependencies for ConversationService
        self.mock_db = mock.MagicMock(spec=Session)
        self.mock_project_service = mock.MagicMock(spec=ProjectService) # Use MagicMock if methods are sync, AsyncMock if async
        self.mock_history_manager = mock.MagicMock(spec=ChatHistoryService)
        self.mock_llm_provider_service = mock.MagicMock(spec=LLMProviderService) # Renamed for clarity
        self.mock_tool_service = mock.MagicMock(spec=ToolService)
        self.mock_prompt_service = mock.MagicMock(spec=PromptService) # Using 'PromptService', will adjust if 'promt_service' is actual
        self.mock_agent_service = mock.MagicMock(spec=AgentsService)
        self.mock_custom_agent_service = mock.MagicMock(spec=CustomAgentService)

        # Make dependent service methods async if they are called with await
        self.mock_project_service.get_project_name = mock.AsyncMock()
        self.mock_agent_service.validate_agent_id = mock.AsyncMock()


        self.user_id_param = "test_user_conv_param" # User ID passed as parameter to create_conversation
        self.user_id_init = "test_user_conv_init"   # User ID for ConversationService constructor
        self.user_email_init = "test_user_conv_init@example.com"

        # Instantiate ConversationService
        # Using 'promt_service' as per instruction if that's the actual param name despite type hint.
        # If PromptService is correct, then param should be prompt_service.
        # The instructions state: "Use actual name ('prompt_service' or 'promt_service')"
        # The previous subtask report said the typo 'promt_service' was intentionally kept.
        self.conversation_service = ConversationService(
            db=self.mock_db,
            user_id=self.user_id_init, 
            user_email=self.user_email_init,
            project_service=self.mock_project_service,
            history_manager=self.mock_history_manager,
            provider_service=self.mock_llm_provider_service, # Use renamed mock
            tools_service=self.mock_tool_service,
            promt_service=self.mock_prompt_service, # Using 'promt_service' as per note
            agent_service=self.mock_agent_service,
            custom_agent_service=self.mock_custom_agent_service
        )

    def tearDown(self):
        self.get_tracer_patcher.stop()

    @mock.patch('app.modules.conversations.conversation.conversation_service.CodeProviderService')
    @mock.patch('app.modules.conversations.conversation.conversation_service.asyncio.create_task') # Mock asyncio.create_task
    async def test_create_conversation_success_creates_span(self, mock_create_task, mock_code_provider_service_cls):
        # Setup
        conv_request = CreateConversationRequest(
            project_ids=["proj_success"], 
            agent_ids=["agent_success"], 
            title="Test Success"
        )
        created_conv_id = "conv_id_success"

        # Mock internal and dependent calls
        self.mock_agent_service.validate_agent_id.return_value = True
        self.mock_project_service.get_project_name.return_value = "Success Project"
        
        # Mock the private method _create_conversation_record
        # It's a regular method on the instance, so assign a MagicMock to it
        self.conversation_service._create_conversation_record = mock.MagicMock(return_value=created_conv_id)
        
        # Mock the private method _add_system_message (it's async)
        self.conversation_service._add_system_message = mock.AsyncMock()
        
        # Mock the CodeProviderService task
        mock_code_provider_instance = mock.MagicMock() # Instance of CodeProviderService
        mock_code_provider_instance.get_project_structure_async = mock.AsyncMock() # Its async method
        mock_code_provider_service_cls.return_value = mock_code_provider_instance # Constructor returns our mock instance


        # Action
        result_conv_id, message = await self.conversation_service.create_conversation(
            conversation_request=conv_request, # Parameter name from instrumentation
            user_id=self.user_id_param, # user_id for the span comes from this param
            hidden=False
        )

        # Assertions
        self.assertEqual(result_conv_id, created_conv_id)
        self.mock_get_tracer.assert_called_once_with('app.modules.conversations.conversation.conversation_service')
        self.mock_tracer.start_as_current_span.assert_called_once_with("conversation.create")
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}
        
        self.assertEqual(actual_attributes.get("user.id"), self.user_id_param) # Should be from param
        self.assertEqual(actual_attributes.get("project.ids"), json.dumps(conv_request.project_ids))
        self.assertEqual(actual_attributes.get("agent.ids"), json.dumps(conv_request.agent_ids))
        self.assertEqual(actual_attributes.get("conversation.id"), created_conv_id)
        self.assertEqual(actual_attributes.get("creation.status"), "success")
        
        self.mock_span.record_exception.assert_not_called()
        
        # Verify mocks
        self.mock_agent_service.validate_agent_id.assert_called_once_with(self.user_id_param, conv_request.agent_ids[0])
        self.conversation_service._create_conversation_record.assert_called_once_with(conv_request, "Success Project", self.user_id_param, False)
        self.conversation_service._add_system_message.assert_called_once_with(created_conv_id, "Success Project", self.user_id_param)
        mock_code_provider_service_cls.assert_called_once_with(self.mock_db)
        mock_create_task.assert_called_once() # Check that asyncio.create_task was called

    @mock.patch('app.modules.conversations.conversation.conversation_service.CodeProviderService')
    @mock.patch('app.modules.conversations.conversation.conversation_service.asyncio.create_task')
    async def test_create_conversation_failure_records_exception(self, mock_create_task, mock_code_provider_service_cls):
        conv_request = CreateConversationRequest(
            project_ids=["proj_fail"], 
            agent_ids=["agent_fail"], 
            title="Test Fail"
        )
        # Simulate a SQLAlchemy IntegrityError
        # The actual error has more specific args, but for testing the type is usually enough
        test_exception = IntegrityError("Simulated DB error", params=None, orig=Exception("original db error"))


        self.mock_agent_service.validate_agent_id.return_value = True
        self.mock_project_service.get_project_name.return_value = "Fail Project"
        
        # Mock _create_conversation_record to raise an exception
        self.conversation_service._create_conversation_record = mock.MagicMock(side_effect=test_exception)
        # _add_system_message and CodeProviderService task won't be called if _create_conversation_record fails early

        # Action & Assert for exception
        # The service wraps IntegrityError in ConversationServiceError
        with self.assertRaises(ConversationServiceError) as context:
            await self.conversation_service.create_conversation(
                conversation_request=conv_request, 
                user_id=self.user_id_param, 
                hidden=False
            )
        
        self.assertIsInstance(context.exception.__cause__, IntegrityError) # Check underlying cause

        # Assertions for telemetry
        self.mock_tracer.start_as_current_span.assert_called_once_with("conversation.create")
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}

        self.assertEqual(actual_attributes.get("user.id"), self.user_id_param)
        self.assertEqual(actual_attributes.get("project.ids"), json.dumps(conv_request.project_ids))
        self.assertEqual(actual_attributes.get("agent.ids"), json.dumps(conv_request.agent_ids))
        # conversation.id is set after _create_conversation_record, which fails here, so it shouldn't be set.
        self.assertNotIn("conversation.id", actual_attributes) 
        self.assertEqual(actual_attributes.get("creation.status"), "failure")
        
        self.mock_span.record_exception.assert_called_once_with(test_exception) # Original exception

if __name__ == '__main__':
    unittest.main()
