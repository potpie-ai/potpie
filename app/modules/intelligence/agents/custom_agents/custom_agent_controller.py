from typing import Any, Dict, List

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    Agent,
    AgentCreate,
    AgentUpdate,
)
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomAgentController:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.service = CustomAgentService(db)

    async def create_agent(self, user_id: str, agent_data: AgentCreate) -> Agent:
        """Create a new custom agent"""
        try:
            return await self.service.create_agent(user_id, agent_data)
        except Exception as e:
            logger.error(f"Error creating custom agent: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create custom agent: {str(e)}",
            )

    async def update_agent(
        self, agent_id: str, user_id: str, agent_data: AgentUpdate
    ) -> Agent:
        """Update an existing custom agent"""
        try:
            agent = await self.service.update_agent(agent_id, user_id, agent_data)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            return agent
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update custom agent: {str(e)}",
            )

    async def delete_agent(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a custom agent"""
        try:
            result = await self.service.delete_agent(agent_id, user_id)
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=result["message"]
                )
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete custom agent: {str(e)}",
            )

    async def get_agent(self, agent_id: str, user_id: str) -> Agent:
        """Get a custom agent by ID"""
        try:
            agent = await self.service.get_agent(agent_id, user_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            return agent
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch custom agent: {str(e)}",
            )


    async def create_agent_from_prompt(
        self,
        prompt: str,
        user_id: str,
    ) -> Agent:
        """Create a custom agent from a natural language prompt"""
        try:
            return await self.service.create_agent_from_prompt(
                prompt=prompt,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Error creating agent from prompt: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent from prompt: {str(e)}",
            )
