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
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentInjectorService:
    def __init__(self, db: Session, provider_service: ProviderService):
        self.sql_db = db
        self.provider_service = provider_service
        self.custom_agent_service = CustomAgentService(db)
        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> Dict[str, Any]:
        return {
            "debugging_agent": DebuggingChatAgent(self.sql_db),
            "codebase_qna_agent": QNAChatAgent(self.sql_db),
            "unit_test_agent": UnitTestAgent(self.sql_db),
            "integration_test_agent": IntegrationTestChatAgent(self.sql_db),
            "code_changes_agent": CodeChangesChatAgent(self.sql_db),
            "LLD_agent": LLDChatAgent(self.sql_db),
            "code_generation_agent": CodeGenerationChatAgent(self.sql_db),
        }

    async def get_system_agent(self, agent_id: str) -> Any:
        """Get an agent instance by ID"""
        if agent_id in self.agents:
            return self.agents[agent_id]
        else:
            return None

    async def validate_agent_id(self, user_id: str, agent_id: str):
        """Validate if an agent ID is valid"""
        return (
            agent_id in self.agents
            or await self.custom_agent_service.get_custom_agent(
                self.sql_db, user_id, agent_id
            )
        )
