from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents_service import AgentInfo
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.agents.chat_agents.adaptive_agent import (
    PromptService,
)
from app.modules.intelligence.tools.tool_service import ToolService


class AgentsController:
    def __init__(
        self,
        db: Session,
        llm_provider: ProviderService,
        prompt_provider: PromptService,
        tools_provider: ToolService,
    ):
        self.service = AgentsService(db, llm_provider, prompt_provider, tools_provider)

    async def list_available_agents(
        self, current_user: dict, list_system_agents: bool
    ) -> List[AgentInfo]:
        try:
            agents = await self.service.list_available_agents(
                current_user, list_system_agents
            )
            return agents
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing agents: {str(e)}"
            )
