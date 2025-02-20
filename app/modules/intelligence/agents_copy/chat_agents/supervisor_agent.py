from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from .auto_router_agent import AutoRouterAgent, AgentWithInfo
from .adaptive_agent import AdaptiveAgent, PromptService, AgentType
from .chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from .system_agents import (
    blast_radius_agent,
    code_gen_agent,
    debug_agent,
    integration_test_agent,
    low_level_design_agent,
    qna_agent,
    unit_test_agent,
)
from typing import AsyncGenerator


class SupervisorAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
        curr_agent_id: str,
    ):

        self.agent = AutoRouterAgent(
            llm_provider,
            agents=[
                AgentWithInfo(
                    id="codebase_qna_agent",
                    name="Codebase Q&A Agent",
                    description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=qna_agent.QnAAgent(llm_provider, tools_provider),
                        agent_type=AgentType.QNA,
                    ),
                ),
                AgentWithInfo(
                    id="debugging_agent",
                    name="Debugging with Knowledge Graph Agent",
                    description="An agent specialized in debugging using knowledge graphs.",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=debug_agent.DebugAgent(llm_provider, tools_provider),
                        agent_type=AgentType.DEBUGGING,
                    ),
                ),
                AgentWithInfo(
                    id="unit_test_agent",
                    name="Unit Test Agent",
                    description="An agent specialized in generating unit tests for code snippets for given function names",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=unit_test_agent.UnitTestAgent(
                            llm_provider, tools_provider
                        ),
                        agent_type=AgentType.UNIT_TEST,
                    ),
                ),
                AgentWithInfo(
                    id="integration_test_agent",
                    name="Integration Test Agent",
                    description="An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=integration_test_agent.IntegrationTestAgent(
                            llm_provider, tools_provider
                        ),
                        agent_type=AgentType.INTEGRATION_TEST,
                    ),
                ),
                AgentWithInfo(
                    id="LLD_agent",
                    name="Low-Level Design Agent",
                    description="An agent specialized in generating a low-level design plan for implementing a new feature.",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=low_level_design_agent.LowLevelDesignAgent(
                            llm_provider, tools_provider
                        ),
                        agent_type=AgentType.LLD,
                    ),
                ),
                AgentWithInfo(
                    id="code_changes_agent",
                    name="Code Changes Agent",
                    description="An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=blast_radius_agent.BlastRadiusAgent(
                            llm_provider, tools_provider
                        ),
                        agent_type=AgentType.CODE_CHANGES,
                    ),
                ),
                AgentWithInfo(
                    id="code_generation_agent",
                    name="Code Generation Agent",
                    description="An agent specialized in generating code for new features or fixing bugs.",
                    agent=AdaptiveAgent(
                        llm_provider,
                        prompt_provider,
                        rag_agent=code_gen_agent.CodeGenAgent(
                            llm_provider, tools_provider
                        ),
                        agent_type=AgentType.CODE_CHANGES,
                    ),
                ),
                # ... Add more here
            ],
            curr_agent_id=curr_agent_id,
        )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self.agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        return await self.agent.run_stream(ctx)
