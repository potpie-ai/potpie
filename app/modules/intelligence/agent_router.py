from app.modules.intelligence.agents.agent_schema import AgentResponse
from app.modules.intelligence.agents.agents_controller import AgentController

from fastapi import APIRouter, Depends
from typing import List
from app.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

class AgentAPI:
    @staticmethod
    @router.get("/agents/", response_model=List[AgentResponse])
    async def list_agents(
        db: Session = Depends(get_db)
    ):
        controller = AgentController(db)
        return await controller.list_agents()
