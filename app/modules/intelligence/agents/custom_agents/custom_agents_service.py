from typing import Dict, Any
from langchain.llms import BaseLLM
from .custom_agent import CustomAgent as CustomAgentExecutor
from .custom_agents_model import CustomAgent
from app.core.database import get_db
from app.modules.intelligence.provider.provider_service import ProviderService

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
        db = next(get_db())
        custom_agent = db.query(CustomAgent).filter(CustomAgent.id == agent_id).first()
        
        if not custom_agent:
            raise ValueError(f"Custom agent with id {agent_id} not found")

        agent_executor = CustomAgentExecutor(self.provider_service.get_llm(), agent_id)
        return agent_executor.run(query)
