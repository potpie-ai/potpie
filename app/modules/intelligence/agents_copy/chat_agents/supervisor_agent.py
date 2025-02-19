from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from .auto_router_agent import AutoRouterAgent, AgentWithInfo
from .adaptive_agent import AdaptiveAgent, PromptService, AgentType
from ..chat_agent import ChatAgent, ChatAgentResponse
from .system_agents import blast_radius_agent, code_gen_agent
from typing import List, Optional, AsyncGenerator


class SupervisorAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
        project_id: str,
    ):

        self.agent = AutoRouterAgent(
            llm_provider,
            agents=[
                AgentWithInfo(
                    id="code_changes_agent",
                    name="Code Changes Agent",
                    description="An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=blast_radius_agent.BlastRadiusAgent(
                            llm_provider, tools_provider, project_id
                        ),
                        agent_type=AgentType.CODE_CHANGES,
                    ),
                ),
                AgentWithInfo(
                    id="code_review_agent",
                    name="Code Review Agent",
                    description="An agent specialized in code reviews, use this for reviewing codes",
                    agent=simple_llm_agent,
                ),
                # ... add all agents here
            ],
            curr_agent_id="code_changes_agent",
        )

    async def run(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        res = await self.run_stream(query, history, node_ids)
        async for response in res:
            return response

        # raise exception if we don't get a response
        raise Exception("response stream failed!!")

    async def run_stream(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        return self.rag_agent.run_stream(query, history, node_ids)
