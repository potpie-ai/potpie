from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agent_execution_controller import (
    AgentExecutionController,
)
from app.modules.intelligence.agents.agent_execution_schema import (
    AgentExecuteRequest,
    AgentExecuteStartResponse,
    AgentExecutionResultResponse,
)

router = APIRouter()


@router.post(
    "/agents/{agent_id}/execute",
    response_model=AgentExecuteStartResponse,
    summary="Start agent execution",
    description="Start a direct agent execution in the background. Returns an execution_id to poll for results.",
)
async def execute_agent(
    agent_id: str,
    request: AgentExecuteRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
) -> AgentExecuteStartResponse:
    user_id: str = user["user_id"]
    controller = AgentExecutionController(db, user_id)
    return await controller.start_execution(agent_id, request)


@router.get(
    "/agents/executions/{execution_id}",
    response_model=AgentExecutionResultResponse,
    summary="Get execution result",
    description="Get the status and result of an agent execution by its execution_id.",
)
async def get_execution_result(
    execution_id: str,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
) -> AgentExecutionResultResponse:
    user_id: str = user["user_id"]
    controller = AgentExecutionController(db, user_id)
    return controller.get_execution_result(execution_id)
