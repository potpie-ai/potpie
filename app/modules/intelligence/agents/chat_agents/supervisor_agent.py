from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .auto_router_agent import AutoRouterAgent
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext, AgentWithInfo
from typing import AsyncGenerator, Dict


class SupervisorAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        agents: Dict[str, AgentWithInfo],
    ):
        self.agent = AutoRouterAgent(llm_provider, agents=agents)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self.agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunck in self.agent.run_stream(ctx):
            yield chunck
