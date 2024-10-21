import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.modules.intelligence.agents.chat_agents.code_changes_agent import (
    CodeChangesAgent,
)
from app.modules.intelligence.agents.chat_agents.debugging_agent import DebuggingAgent
from app.modules.intelligence.agents.chat_agents.integration_test_agent import (
    IntegrationTestAgent,
)

from app.modules.intelligence.agents.chat_agents.lld_agent import LLDAgent
from app.modules.intelligence.agents.chat_agents.qna_agent import QNAAgent
from app.modules.intelligence.agents.chat_agents.unit_test_agent import UnitTestAgent
from app.modules.intelligence.provider.provider_service import ProviderService

logger = logging.getLogger(__name__)


class AgentInjectorService:
    def __init__(self, db: Session, provider_service: ProviderService):
        self.sql_db = db
        self.provider_service = provider_service
        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> Dict[str, Any]:
        mini_llm = self.provider_service.get_small_llm()
        reasoning_llm = self.provider_service.get_large_llm()
        return {
            "debugging_agent": DebuggingAgent(mini_llm, reasoning_llm, self.sql_db),
            "codebase_qna_agent": QNAAgent(mini_llm, reasoning_llm, self.sql_db),
            "unit_test_agent": UnitTestAgent(mini_llm, reasoning_llm, self.sql_db),
            "integration_test_agent": IntegrationTestAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
            "code_changes_agent": CodeChangesAgent(
                mini_llm, reasoning_llm, self.sql_db
            ),
            "LLD_agent": LLDAgent(mini_llm, reasoning_llm, self.sql_db),
        }

    def get_agent(self, agent_id: str) -> Any:
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Invalid agent_id: {agent_id}")
            raise ValueError(f"Invalid agent_id: {agent_id}")
        return agent

    def validate_agent_id(self, agent_id: str) -> bool:
        logger.info(f"Validating agent_id: {agent_id}")
        return agent_id in self.agents
