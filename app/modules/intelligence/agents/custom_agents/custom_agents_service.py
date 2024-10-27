import httpx
import logging
from typing import Dict, Any, List

from app.modules.conversations.message.message_schema import NodeContext

logger = logging.getLogger(__name__)

class CustomAgentsService:
    def __init__(self):
        self.base_url = "https://your-custom-agent-service-url.com"  # Replace with actual URL

    async def run_agent(
        self,
        agent_id: str,
        query: str,
        project_id: str,
        user_id: str,
        node_ids: List[NodeContext],
    ) -> Dict[str, Any]:
        run_url = f"{self.base_url}/api/v1/agents/{agent_id}/run"
        payload = {
            "query": query,
            "project_id": project_id,
            "user_id": user_id,
            "node_ids": [node.dict() for node in node_ids],
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(run_url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred while running agent {agent_id}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error occurred while running agent {agent_id}: {e}")
                raise

    async def validate_agent(self, agent_id: str) -> bool:
        return True
        # validate_url = f"{self.base_url}/api/v1/agents/{agent_id}/validate"

        # async with httpx.AsyncClient() as client:
        #     try:
        #         response = await client.get(validate_url)
        #         return response.status_code == 200
        #     except Exception as e:
        #         logger.error(f"Error validating agent {agent_id}: {e}")
        #         return False
