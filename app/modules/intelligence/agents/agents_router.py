from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.agents.agent_list_scope import AgentListMode
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.agents.agents_service import AgentInfo
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.prompts.prompt_service import PromptService
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
        mode: AgentListMode = Query(
            default=AgentListMode.RUNTIME,
            description=(
                "Server-defined list scope. "
                "'runtime' = system + owned + shared-with-caller; "
                "'owned' = only agents owned by the caller. "
                "Legacy list_system_agents/include_public/include_shared are ignored."
            ),
        ),
        # Accepted for backward compatibility only — never used for authorization.
        list_system_agents: Optional[bool] = Query(
            default=None,
            description="Deprecated. Ignored; use mode instead.",
            include_in_schema=False,
        ),
        include_public: Optional[bool] = Query(
            default=None,
            description="Deprecated. Ignored; use mode instead.",
            include_in_schema=False,
        ),
        include_shared: Optional[bool] = Query(
            default=None,
            description="Deprecated. Ignored; use mode instead.",
            include_in_schema=False,
        ),
    ):
        # Bind legacy params so FastAPI accepts them, then discard (FW003).
        del list_system_agents, include_public, include_shared

        user_id: str = user["user_id"]
        llm_provider = ProviderService(db, user_id)
        tools_provider = ToolService(db, user_id)
        prompt_provider = PromptService(db)
        controller = AgentsController(db, llm_provider, prompt_provider, tools_provider)
        return await controller.list_available_agents(user, mode)
