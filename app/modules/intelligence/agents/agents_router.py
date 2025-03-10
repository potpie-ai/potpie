from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.agents.agents_service import AgentInfo
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.agents.chat_agents.adaptive_agent import (
    PromptService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.agents.custom_agents.custom_agent_router import (
    router as custom_agent_router,
)

router = APIRouter()

# Include custom agent router with the original path
router.include_router(
    custom_agent_router, prefix="/custom-agents/agents", tags=["Custom Agents"]
)


class AgentsAPI:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    @router.get("/list-available-agents/", response_model=List[AgentInfo])
    async def list_available_agents(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
        list_system_agents: bool = Query(
            default=True, description="Include system agents in the response"
        ),
    ):
        user_id: str = user["user_id"]
        llm_provider = ProviderService(db, user_id)
        tools_provider = ToolService(db, user_id)
        prompt_provider = PromptService(db)
        controller = AgentsController(db, llm_provider, prompt_provider, tools_provider)
        return await controller.list_available_agents(user, list_system_agents)
