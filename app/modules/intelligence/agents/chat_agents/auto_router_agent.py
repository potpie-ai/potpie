from typing import AsyncGenerator, Dict
from app.modules.intelligence.agents.chat_agent import (
    ChatAgentResponse,
    ChatContext,
    ChatAgent,
    AgentWithInfo,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)


class AutoRouterAgent(ChatAgent):
    """AutoRouterAgent returns the current agent from the context without routing."""

    def __init__(
        self,
        llm_provider: ProviderService,
        agents: Dict[str, AgentWithInfo],
    ):
        self.llm_provider = llm_provider
        self.agents = agents

    async def _get_current_agent(self, ctx: ChatContext) -> ChatAgent:
        """Returns the agent corresponding to the current agent ID in the context."""
        return self.agents[ctx.curr_agent_id].agent

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._get_current_agent(ctx)
        return await agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._get_current_agent(ctx)
        async for chunk in agent.run_stream(ctx):
            yield chunk
