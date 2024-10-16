from typing import Dict, Any

from app.modules.intelligence.agents.custom_agents.custom_agents_model import CustomAgent
from .custom_agent import CustomAgent as CustomAgentExecutor
from app.modules.intelligence.provider.provider_service import ProviderService
from sqlalchemy.orm import Session

class CustomAgentService:
    def __init__(self, db: Session, provider_service: ProviderService):
        self.db = db
        self.provider_service = provider_service

    async def get_custom_agent(self, agent_id: str) -> CustomAgent:
        llm = self.provider_service.get_llm()
        return CustomAgent(llm, agent_id)

    def is_custom_agent(self, agent_id: str) -> bool:
        # Implement logic to check if the agent_id corresponds to a custom agent
        # This could involve checking a database table or a predefined list
        pass

    async def execute_custom_agent(self, agent_id: str, query: str) -> str:
        agent_executor = CustomAgentExecutor(self.provider_service.get_llm(), agent_id)
        return agent_executor.run(query)
