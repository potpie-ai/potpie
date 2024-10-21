import httpx
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Union
from app.modules.conversations.message.message_schema import NodeContext

class CustomAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.base_url = "http://localhost:8080"

    async def run(
        self,
        agent_id: str,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: Union[List[NodeContext], List[str]]
    ) -> str:
        # Import CustomAgent here to avoid circular import
        from app.modules.intelligence.agents.custom_agents.custom_agent import CustomAgent
        
        custom_agent = CustomAgent(agent_id)
        
        # Convert node_ids to a list of dictionaries or strings
        node_ids_payload = [
            node.dict() if isinstance(node, NodeContext) else node
            for node in node_ids
        ]
        
        payload = {
            "query": query,
            "project_id": project_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "node_ids": node_ids_payload
        }
        
        return await custom_agent.run(payload)

    # async def get_system_prompt(self, agent_id: str) -> str:
    #     system_prompt_url = f"{self.base_url}/deployment/{agent_id}/system_prompt"
        
    #     async with httpx.AsyncClient() as client:
    #         response = await client.get(system_prompt_url)
    #         response.raise_for_status()
    #         return response.text

    async def is_valid_agent(self, agent_id: str) -> bool:
        # Import CustomAgent here to avoid circular import
        from app.modules.intelligence.agents.custom_agents.custom_agent import CustomAgent
        
        custom_agent = CustomAgent(agent_id)
        return await custom_agent.is_valid()
