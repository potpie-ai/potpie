import logging
import os
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.agents_service import AgentsService

logger = logging.getLogger(__name__)

load_dotenv()


class CustomAgentsService:
    def __init__(self):
        self.base_url = os.getenv("POTPIE_PLUS_BASE_URL")

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

        # 1200 seconds = 20 minutes timeout for the entire request
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=1200.0)) as client:
            try:
                response = await client.post(run_url, json=payload)
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

    async def validate_agent(self,db: Session, user_id: str, agent_id: str) -> bool:
        try:
            agents_service = AgentsService(db)
            custom_agents = await agents_service.fetch_custom_agents(user_id)
            return any(agent.id == agent_id for agent in custom_agents)
        except Exception as e:
            logger.error(f"Error validating agent {agent_id}: {str(e)}")
            return False
