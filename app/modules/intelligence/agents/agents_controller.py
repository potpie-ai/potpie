from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents_schema import Agent, AgentInfo
from app.modules.intelligence.agents.agents_service import AgentsService


class AgentsController:
    def __init__(self, db: Session):
        self.service = AgentsService(db)  # Direct instantiation

    async def list_available_agents(self) -> List[AgentInfo]:
        try:
            agents = await self.service.list_available_agents()
            return agents
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing agents: {str(e)}"
            )

    async def create_custom_agent(
        self,
        user_id: str,
        role: str,
        goal: str,
        backstory: str,
        tool_ids: List[str],
        tasks: List[dict],
    ) -> Agent:
        return await self.service.create_custom_agent(
            user_id=user_id,
            role=role,
            goal=goal,
            backstory=backstory,
            tool_ids=tool_ids,
            tasks=tasks,
        )
