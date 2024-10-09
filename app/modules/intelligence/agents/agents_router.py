from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.agents.agents_schema import Agent, AgentCreate, AgentInfo

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
    async def create_or_update_agent(
        request: AgentCreate,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = AgentsController(db)
        try:
            return await controller.create_or_update_agent(
                user_id=user_id,
                role=request.role,
                goal=request.goal,
                backstory=request.backstory,
                tool_ids=request.tool_ids,
                tasks=[task.model_dump(exclude_unset=True) for task in request.tasks],
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
