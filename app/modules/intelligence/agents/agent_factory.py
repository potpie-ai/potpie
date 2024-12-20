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


class AgentFactory:
    def __init__(self, db: Session, provider_service: ProviderService):
        self.db = db
        self.provider_service = provider_service
        self._agent_cache: Dict[str, Any] = {}

    def get_agent(self, agent_id: str, user_id: str) -> Any:
        """Get or create an agent instance"""
        cache_key = f"{agent_id}_{user_id}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        mini_llm = self.provider_service.get_small_llm(agent_type=AgentType.LANGCHAIN)
        reasoning_llm = self.provider_service.get_large_llm(
            agent_type=AgentType.LANGCHAIN
        )

        agent = self._create_agent(agent_id, mini_llm, reasoning_llm, user_id)
        self._agent_cache[cache_key] = agent
        return agent

    def _create_agent(
        self, agent_id: str, mini_llm, reasoning_llm, user_id: str
    ) -> Any:
        """Create a new agent instance"""
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

        if agent_id in agent_map:
            return agent_map[agent_id]()

        # If not a system agent, create custom agent
        return CustomAgent(
            llm=reasoning_llm, db=self.db, agent_id=agent_id, user_id=user_id
        )
