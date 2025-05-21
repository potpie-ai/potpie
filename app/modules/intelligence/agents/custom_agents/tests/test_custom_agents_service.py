import unittest
from unittest import mock
import asyncio # For async tests

# Actual imports for CustomAgentService and its dependencies/schemas
from app.modules.intelligence.agents.custom_agents.custom_agents_service import CustomAgentService
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import AgentCreate, Agent as AgentSchema, TaskCreate # Agent for return type
from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent as CustomAgentModel # SQLAlchemy model
from sqlalchemy.exc import SQLAlchemyError # For simulating DB errors
from sqlalchemy.orm import Session # For type hinting Session

# Need to ensure app.core.telemetry.get_tracer is available for patching
import app.core.telemetry 

class TestCustomAgentServiceTelemetry(unittest.IsolatedAsyncioTestCase): # Use IsolatedAsyncioTestCase for async methods

    def setUp(self):
        self.mock_tracer = mock.MagicMock()
        self.mock_span = mock.MagicMock()
        # Configure the context manager mock for the span
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span
        self.mock_tracer.start_as_current_span.return_value.__exit__ = mock.MagicMock(return_value=None)

        # Patch get_tracer for the custom_agents_service module
        self.get_tracer_patcher = mock.patch(
            'app.modules.intelligence.agents.custom_agents.custom_agents_service.get_tracer', 
            return_value=self.mock_tracer
        )
        self.mock_get_tracer = self.get_tracer_patcher.start()

        self.mock_db = mock.MagicMock(spec=Session)
        
        # Instantiate CustomAgentService
        self.custom_agent_service = CustomAgentService(db=self.mock_db)
        self.user_id_for_test = "custom_agent_user_1"

        # Mock methods called by create_agent
        # fetch_available_tools is async
        self.custom_agent_service.fetch_available_tools = mock.AsyncMock(return_value=["tool1", "tool2"])
        # enhance_task_descriptions is async
        self.custom_agent_service.enhance_task_descriptions = mock.AsyncMock(return_value=[{"description": "enhanced_desc", "tools": [], "expected_output": {}}])
        # persist_agent is sync and returns an AgentSchema object
        self.custom_agent_service.persist_agent = mock.MagicMock()


    def tearDown(self):
        self.get_tracer_patcher.stop()

    async def test_create_agent_success_creates_span(self):
        # Setup
        agent_create_data = AgentCreate(
            role="TestRole", 
            goal="Test Goal", # Added goal
            backstory="Test Backstory", # Added backstory
            tasks=[TaskCreate(description="Test Task", tools=[])] # Tasks need to be TaskCreate
        ) 
        
        # Mock what persist_agent returns (an Agent schema object)
        # The actual ID for the span comes from this returned object.
        persisted_agent_schema = AgentSchema(
            id="agent_id_123", 
            user_id=self.user_id_for_test,
            role="TestRole", 
            goal="Test Goal",
            backstory="Test Backstory",
            tasks=[], # Simplified for test
            deployment_url=None,
            created_at=mock.MagicMock(), # Mock datetime
            updated_at=mock.MagicMock(), # Mock datetime
            visibility="PRIVATE" # Default or example visibility
        )
        self.custom_agent_service.persist_agent.return_value = persisted_agent_schema

        # Action
        returned_agent = await self.custom_agent_service.create_agent(
            user_id=self.user_id_for_test, 
            agent_data=agent_create_data
        )

        # Assertions
        self.mock_get_tracer.assert_called_once_with('app.modules.intelligence.agents.custom_agents.custom_agents_service')
        self.mock_tracer.start_as_current_span.assert_called_once_with("custom_agent.create")
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}
        
        self.assertEqual(actual_attributes.get("user.id"), self.user_id_for_test)
        self.assertEqual(actual_attributes.get("agent.name"), agent_create_data.role) 
        self.assertEqual(actual_attributes.get("agent.role"), agent_create_data.role)
        self.assertEqual(actual_attributes.get("agent.id"), str(persisted_agent_schema.id)) 
        self.assertEqual(actual_attributes.get("creation.status"), "success")
        
        self.mock_span.record_exception.assert_not_called()
        
        # Verify persist_agent was called correctly
        # enhance_task_descriptions is called before persist_agent
        enhanced_tasks_mock_return = await self.custom_agent_service.enhance_task_descriptions.return_value
        self.custom_agent_service.persist_agent.assert_called_once_with(
            self.user_id_for_test, agent_create_data, enhanced_tasks_mock_return
        )

    async def test_create_agent_failure_records_exception(self):
        agent_create_data = AgentCreate(
            role="FailRole", 
            goal="Fail Goal",
            backstory="Fail Backstory",
            tasks=[TaskCreate(description="Fail Task", tools=[])]
        )
        test_exception = SQLAlchemyError("Simulated DB error from persist_agent")

        # Mock persist_agent to raise an exception
        self.custom_agent_service.persist_agent.side_effect = test_exception
        
        # Action & Assert for exception
        # The create_agent method has a try/except SQLAlchemyError that re-raises HTTPException
        # However, the span should record the original SQLAlchemyError.
        # The prompt asks to check for SQLAlchemyError for the span, but the method might raise HTTPException.
        # Let's catch HTTPException as that's what the user of create_agent would see.
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as context:
            await self.custom_agent_service.create_agent(
                user_id=self.user_id_for_test, 
                agent_data=agent_create_data
            )
        
        self.assertEqual(context.exception.status_code, 500) # As it's re-raised

        # Assertions for telemetry
        self.mock_tracer.start_as_current_span.assert_called_once_with("custom_agent.create")
        
        actual_attributes = {call[0][0]: call[0][1] for call in self.mock_span.set_attribute.call_args_list}

        self.assertEqual(actual_attributes.get("user.id"), self.user_id_for_test)
        self.assertEqual(actual_attributes.get("agent.name"), agent_create_data.role)
        self.assertEqual(actual_attributes.get("agent.role"), agent_create_data.role)
        self.assertNotIn("agent.id", actual_attributes) # Error before agent ID is available from persist_agent
        self.assertEqual(actual_attributes.get("creation.status"), "failure")
        
        self.mock_span.record_exception.assert_called_once_with(test_exception) # Original exception
        
        # Verify DB rollback is called by the except block in create_agent
        self.mock_db.rollback.assert_called_once()

if __name__ == '__main__':
    asyncio.run(unittest.main())
