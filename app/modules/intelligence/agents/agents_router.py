from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.agents.agents_schema import AgentInfo

router = APIRouter()


class AgentsAPI:
    @staticmethod
    @router.get("/list-available-agents/", response_model=List[AgentInfo])
    async def list_available_agents(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        controller = AgentsController(db)
        return await controller.list_available_agents()

    @staticmethod
    @router.get("/agent-details/{agent_id}", response_model=AgentInfo)
    async def get_agent_details(
        agent_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        controller = AgentsController(db)
        try:
            return await controller.get_agent_details(agent_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

