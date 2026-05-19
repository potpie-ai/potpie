from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
    AgentType as MultiAgentType,
)
from app.modules.intelligence.agents.chat_agents.multi_agent.agent_factory import (
    create_integration_agents,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator, Optional
from observability import get_logger
from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent_prompt import (
    debug_task_prompt,
)

logger = get_logger(__name__)


class DebugAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.tools_provider = tools_provider
        self.llm_provider = llm_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self, ctx: Optional[ChatContext] = None) -> ChatAgent:
        agent_config = AgentConfig(
            role="Debugging and Code Analysis Specialist",
            goal="Provide comprehensive debugging solutions and code analysis by identifying root causes, tracing code flows, and delivering precise fixes. For general queries, maintain a conversational approach while grounding responses in code context.",
            backstory="""
                    You are a seasoned debugging engineer with deep expertise in systematic problem-solving, root cause analysis, and code comprehension. You excel at:
                    1. Conversational code exploration and Q&A - helping users understand codebases naturally
                    2. Systematic debugging - when faced with bugs, you follow rigorous methodologies to find root causes
                    3. Strategic thinking - you fix problems at their source, not just patch symptoms
                    4. Code navigation - you expertly traverse knowledge graphs, code structures, and relationships
                    5. Contextual understanding - you build comprehensive mental models of how code fits together

                    You adapt your approach: conversational for questions, methodical for debugging. You use todo and requirements tools to track progress and ensure thoroughness.
                """,
            tasks=[
                TaskConfig(
                    description=debug_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results. For debugging tasks, includes root cause analysis, fix location rationale, and implementation details.",
                )
            ],
        )

        # Exclude embedding-dependent tools during INFERRING status
        exclude_embedding_tools = ctx.is_inferring() if ctx else False
        if exclude_embedding_tools:
            logger.info(
                "Project is in INFERRING status - excluding embedding-dependent tools"
            )

        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "query_context_graph",
                "ask_knowledge_graph_queries",
                "get_nodes_from_tags",
                "get_code_file_structure",
                "search_text",
                "get_workspace_debug_context",
                "parse_failure_signal",
                "run_validation",
                # DAP debugger tools (A3)
                "start_debug_session",
                "set_breakpoints",
                "take_debug_snapshot",
                "step_over",
                "step_into",
                "step_out",
                "continue_execution",
                "evaluate_expression",
                "list_debug_sessions",
                "stop_debug_session",
                "webpage_extractor",
                "web_search_tool",
                "fetch_file",
                "fetch_files_batch",
                "analyze_code_structure",
                "sandbox_text_editor",
                "sandbox_shell",
                "sandbox_search",
                "sandbox_git",
                "read_todos",
                "write_todos",
                "add_todo",
                "update_todo_status",
                "remove_todo",
                "add_subtask",
                "set_dependency",
                "get_available_tasks",
                "add_requirements",
                "get_requirements",
                "record_hypothesis",
                "update_hypothesis_status",
                "append_hypothesis_evidence",
                "list_hypotheses",
            ],
            exclude_embedding_tools=exclude_embedding_tools,
        )

        supports_pydantic = self.llm_provider.supports_pydantic("chat")
        should_use_multi = MultiAgentConfig.should_use_multi_agent("debugging_agent")

        logger.info(
            "DebugAgent routing",
            supports_pydantic=supports_pydantic,
            should_use_multi=should_use_multi,
        )

        if supports_pydantic:
            if should_use_multi:
                logger.info("✅ Using PydanticMultiAgent (multi-agent system)")
                # Create specialized delegate agents for debugging: THINK_EXECUTE + integration agents
                integration_agents = create_integration_agents()
                delegate_agents = {
                    MultiAgentType.THINK_EXECUTE: AgentConfig(
                        role="Debug Solution Specialist",
                        goal="Provide comprehensive debugging solutions",
                        backstory="Expert at creating debugging strategies and solutions.",
                        tasks=[
                            TaskConfig(
                                description="Create debugging solutions and strategies",
                                expected_output="Debugging solution plan",
                            )
                        ],
                        max_iter=12,
                    ),
                    **integration_agents,
                }
                return PydanticMultiAgent(
                    self.llm_provider,
                    agent_config,
                    tools,
                    None,
                    delegate_agents,
                    tools_provider=self.tools_provider,
                )
            else:
                logger.info("❌ Multi-agent disabled by config, using PydanticRagAgent")
                return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            logger.error(
                "❌ Model does not support Pydantic - using fallback PydanticRagAgent"
            )
            return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code referred to in the query:\n {code_results}\n"
            )
        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        ctx = await self._enriched_context(ctx)
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent(ctx).run_stream(ctx):
            yield chunk


# debug_task_prompt is imported from debug_agent_prompt above.
# It is defined there to keep the prompt importable without pulling in the full
# agent stack (provider_service → botocore, etc.), enabling lightweight smoke tests.
