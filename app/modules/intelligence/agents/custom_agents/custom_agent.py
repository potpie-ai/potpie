# import json
# from typing import Dict, List, Any
# from sqlalchemy.orm import Session

# from app.modules.conversations.message.message_schema import NodeContext
# from app.modules.intelligence.agents.custom_agents.agent_executor import AgentExecutor
# from app.modules.utils.logger import setup_logger

# logger = setup_logger(__name__)

# class CustomAgent:
#     """A custom agent that can be configured with a specific role, goal, and system prompt."""

#     def __init__(self, llm: Any, db: Session, system_prompt: str, user_id: str):
#         self.llm = llm
#         self.db = db
#         self.user_id = user_id
#         self.system_prompt = system_prompt
#         self.executor = AgentExecutor(llm, db, system_prompt, user_id)

#     async def run(
#         self,
#         query: str,
#         project_id: str,
#         conversation_id: str,
#         node_ids: List[NodeContext],
#     ) -> Dict[str, Any]:
#         """Run the agent with the given query"""
#         try:
#             full_response = ""
#             async for chunk in self.executor.run(
#                 query=query,
#                 project_id=project_id,
#                 conversation_id=conversation_id,
#                 node_ids=node_ids
#             ):
#                 response_data = json.loads(chunk)
#                 full_response += response_data["message"]

#             return {
#                 "response": full_response,
#                 "conversation_id": conversation_id
#             }
#         except Exception as e:
#             logger.error(f"Error running custom agent: {str(e)}")
#             raise

import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.custom_agents.agent_executor import AgentExecutor
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()


class CustomAgent:
    def __init__(self, llm: Any, db: Session, agent_id: str, user_id: str):
        self.llm = llm
        self.db = db
        self.agent_id = agent_id
        self.user_id = user_id
        self.executor = None

    async def _get_system_prompt(self) -> str:
        """Fetch system prompt using the get_agent method"""
        from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
            CustomAgentService,
        )

        agent = await CustomAgentService(self.db).get_agent(self.agent_id, self.user_id)
        if not agent:
            raise ValueError(f"Agent {self.agent_id} not found for user {self.user_id}")
        return agent.system_prompt

    async def _initialize_executor(self) -> None:
        """Initialize the agent executor with the system prompt"""
        if not self.executor:
            system_prompt = await self._get_system_prompt()
            if not system_prompt:
                raise ValueError(f"System prompt not found for agent {self.agent_id}")
            self.executor = AgentExecutor(
                self.llm, self.db, system_prompt, self.user_id, agent_id=self.agent_id
            )

    async def run(
        self,
        agent_id: str,
        query: str,
        project_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> Dict[str, Any]:
        """Run the agent with the given query"""
        try:
            await self._initialize_executor()

            full_response = ""
            async for chunk in self.executor.run(
                agent_id=agent_id,
                query=query,
                project_id=project_id,
                conversation_id=conversation_id,
                node_ids=node_ids,
            ):
                response_data = json.loads(chunk)
                full_response += response_data["message"]

            return {"response": full_response, "conversation_id": conversation_id}
        except Exception as e:
            logger.error(f"Error running custom agent: {str(e)}")
            raise
