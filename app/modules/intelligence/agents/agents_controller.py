from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents_schema import AgentInfo
from app.modules.intelligence.agents.agents_service import AgentsService


class AgentsController:
    def __init__(self, db: Session):
        self.service = AgentsService.create(db)

    async def list_available_agents(self) -> List[AgentInfo]:
        try:
            agents = await self.service.list_available_agents()
            return agents
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing agents: {str(e)}"
            )
