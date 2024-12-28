import asyncio
import logging
from typing import Any, Dict
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.chat_agents.code_changes_chat_agent import (
    CodeChangesChatAgent,
)
from app.modules.intelligence.agents.chat_agents.code_gen_chat_agent import (
    CodeGenerationChatAgent,
)
from app.modules.intelligence.agents.chat_agents.debugging_chat_agent import (
    DebuggingChatAgent,
)
from app.modules.intelligence.agents.chat_agents.integration_test_chat_agent import (
    IntegrationTestChatAgent,
)
from app.modules.intelligence.agents.chat_agents.lld_chat_agent import LLDChatAgent
from app.modules.intelligence.agents.chat_agents.qna_chat_agent import QNAChatAgent
from app.modules.intelligence.agents.chat_agents.unit_test_chat_agent import (
    UnitTestAgent,
)
from app.modules.intelligence.agents.custom_agents.custom_agent import CustomAgent
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.utils.rate_limiter import RateLimiter

# Configure logging
logger = logging.getLogger(__name__)

class AgentFactory:
    def __init__(self, db: Session, provider_service: ProviderService):
        """Initialize AgentFactory with database session and provider service."""
        logger.info("Initializing AgentFactory")
        self.db = db
        self.provider_service = provider_service
        self._agent_cache: Dict[str, Any] = {}
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter("agent")
        logger.debug("Rate limiter initialized for agent factory")

    async def get_agent(self, agent_id: str, user_id: str) -> Any:
        """
        Get or create an agent instance.
        
        Args:
            agent_id (str): The identifier for the agent type
            user_id (str): The user's identifier
            
        Returns:
            Any: The requested agent instance
        """
        logger.info(f"Getting agent for agent_id: {agent_id}, user_id: {user_id}")
        
        try:
            await self.rate_limiter.acquire()
            cache_key = f"{agent_id}_{user_id}"

            if cache_key in self._agent_cache:
                logger.debug(f"Retrieved agent from cache for key: {cache_key}")
                return self._agent_cache[cache_key]

            logger.debug("Initializing LLMs for new agent")
            mini_llm = await self.provider_service.get_small_llm(agent_type=AgentType.LANGCHAIN)
            reasoning_llm = await self.provider_service.get_large_llm(
                agent_type=AgentType.LANGCHAIN
            )

            logger.debug(f"Creating new agent instance for agent_id: {agent_id}")
            agent = self._create_agent(agent_id, mini_llm, reasoning_llm, user_id)
            self._agent_cache[cache_key] = agent
            logger.info(f"Successfully created and cached agent for key: {cache_key}")
            return agent
            
        except Exception as e:
            logger.error(f"Error getting agent: {str(e)}", exc_info=True)
            raise

    def _create_agent(
        self, agent_id: str, mini_llm, reasoning_llm, user_id: str
    ) -> Any:
        """
        Create a new agent instance based on the agent_id.
        
        Args:
            agent_id (str): The identifier for the agent type
            mini_llm: The small language model instance
            reasoning_llm: The large language model instance
            user_id (str): The user's identifier
            
        Returns:
            Any: The created agent instance
        """
        logger.debug(f"Creating agent with id: {agent_id}")
        
        agent_map = {
            "debugging_agent": lambda: DebuggingChatAgent(
                mini_llm, reasoning_llm, self.db
            ),
            "codebase_qna_agent": lambda: QNAChatAgent(
                mini_llm, reasoning_llm, self.db
            ),
            "unit_test_agent": lambda: UnitTestAgent(mini_llm, reasoning_llm, self.db),
            "integration_test_agent": lambda: IntegrationTestChatAgent(
                mini_llm, reasoning_llm, self.db
            ),
            "code_changes_agent": lambda: CodeChangesChatAgent(
                mini_llm, reasoning_llm, self.db
            ),
            "LLD_agent": lambda: LLDChatAgent(mini_llm, reasoning_llm, self.db),
            "code_generation_agent": lambda: CodeGenerationChatAgent(
                mini_llm, reasoning_llm, self.db
            ),
        }

        try:
            if agent_id in agent_map:
                logger.info(f"Creating system agent: {agent_id}")
                return agent_map[agent_id]()

            # If not a system agent, create custom agent
            logger.info(f"Creating custom agent for agent_id: {agent_id}")
            return CustomAgent(
                llm=reasoning_llm, db=self.db, agent_id=agent_id, user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error creating agent {agent_id}: {str(e)}", exc_info=True)
            raise