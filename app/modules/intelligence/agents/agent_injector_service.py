import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.modules.intelligence.agents.chat_agents.code_changes_chat_agent import (
    CodeChangesChatAgent,
)
from app.modules.intelligence.agents.chat_agents.debugging_chat_agent import DebuggingChatAgent
from app.modules.intelligence.agents.chat_agents.integration_test_chat_agent import (
    IntegrationTestChatAgent,
)
from app.modules.intelligence.agents.chat_agents.lld_chat_agent import LLDChatAgent
from app.modules.intelligence.agents.chat_agents.qna_chat_agent import QNAChatAgent
from app.modules.intelligence.agents.chat_agents.unit_test_chat_agent import UnitTestAgent
from app.modules.intelligence.agents.custom_agents.custom_agent import CustomAgent
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentsService,
)
from app.modules.intelligence.provider.provider_service import ProviderService

logger = logging.getLogger(__name__)


class AgentInjectorService:
    def __init__(self, db: Session, provider_service: ProviderService, user_id: str):
        self.sql_db = db
        self.provider_service = provider_service
        self.custom_agent_service = CustomAgentsService()
        self.agents = self._initialize_agents()
        self.user_id = user_id

    def _initialize_agents(self) -> Dict[str, Any]:
        mini_llm = self.provider_service.get_small_llm()
        reasoning_llm = self.provider_service.get_large_llm()
        return {
            "debugging_agent": DebuggingChatAgent(mini_llm, reasoning_llm, self.sql_db),
            "codebase_qna_agent": QNAChatAgent(mini_llm, reasoning_llm, self.sql_db),
            "unit_test_agent": UnitTestAgent(mini_llm, reasoning_llm, self.sql_db),
            "integration_test_agent": IntegrationTestChatAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
            "code_changes_agent": CodeChangesChatAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
            "LLD_agent": LLDChatAgent(mini_llm, reasoning_llm, self.sql_db),
        }

    def get_agent(self, agent_id: str) -> Any:
        if agent_id in self.agents:
            return self.agents[agent_id]
        else:
            reasoning_llm = self.provider_service.get_large_llm()
            return CustomAgent(
                llm=reasoning_llm,
                db=self.sql_db,
                agent_id=agent_id,
                user_id=self.user_id,
            )

    def validate_agent_id(self, user_id: str, agent_id: str) -> bool:
        return agent_id in self.agents or self.custom_agent_service.validate_agent(
            self.sql_db, user_id, agent_id
        )
