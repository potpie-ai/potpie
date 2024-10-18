import httpx
from sqlalchemy.orm import Session
from typing import Dict, List
from app.modules.conversations.message.message_schema import NodeContext

class CustomAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.base_url = "http://localhost:8080"

    async def execute_custom_agent(
        self, 
        agent_id: str, 
        query: str, 
        node_ids: List[NodeContext]
    ) -> Dict[str, str]:
        deployment_url = f"{self.base_url}/deployment/{agent_id}/query"
        
        payload = {
            "query": query,
            "node_ids": [node.dict() for node in node_ids]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(deployment_url, json=payload)
            response.raise_for_status()
            return response.json()

    async def get_system_prompt(self, agent_id: str) -> str:
        system_prompt_url = f"{self.base_url}/deployment/{agent_id}/system_prompt"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(system_prompt_url)
            response.raise_for_status()
            return response.text
