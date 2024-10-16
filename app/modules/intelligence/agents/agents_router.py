from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService, get_current_user
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.agents.agents_schema import Agent, AgentCreate, AgentInfo
from app.modules.intelligence.agents.custom_agents.custom_agents_service import CustomAgentService

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
    @router.post("/agents/", response_model=Agent)
    async def create_custom_agent(
        request: AgentCreate,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        custom_agent_service = CustomAgentService(db)
        try:
            return await custom_agent_service.create_custom_agent(
                user_id=user_id,
                role=request.role,
                goal=request.goal,
                backstory=request.backstory,
                tasks=[task.model_dump(exclude_unset=True) for task in request.tasks],
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
