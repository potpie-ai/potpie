import logging
from typing import Any, Dict, List

import httpx

from app.modules.conversations.message.message_schema import NodeContext

logger = logging.getLogger(__name__)


class CustomAgentsService:
    def __init__(self):
        self.base_url = "http://localhost:8000"  # Replace with actual URL

    async def run_agent(
        self,
        agent_id: str,
        query: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext] = None,
    ) -> Dict[str, Any]:
        run_url = f"{self.base_url}/custom-agents/agents/{agent_id}/query"
        payload = {
            "user_id": user_id,
            "query": query,
            "conversation_id": conversation_id,
        }

        if node_ids:
            payload["node_ids"] = [node.dict() for node in node_ids]

        # Set a reasonable timeout of 10 minutes to avoid indefinite waits
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600.0)) as client:
            try:
                response = await client.post(run_url, json=payload)
                print("response from agent service", response.json())
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.error(
                    f"Request timed out after 10 minutes while running agent {agent_id}: {e}"
                )
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred while running agent {agent_id}: {e}")
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error occurred while running agent {agent_id}: {e}"
                )
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
