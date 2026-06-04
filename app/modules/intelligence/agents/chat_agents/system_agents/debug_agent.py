from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_deep_debug_agent import (
    PydanticDeepDebugAgent,
)
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

# Base tools always requested for the DebugAgent (per the debug-agent spec).
# Keep this list focused: file/text navigation, workspace context, failure-signal
# parsing, hypothesis tracking, focused todo ops, and validation.
# Excludes (deliberately): KG/code-graph tools, web tools, broad sandbox shell /
# edit / git tools, broad todo tools (write_todos, remove_todo, add_subtask,
# set_dependency), and requirements tools — these are out of scope for the
# debugging agent.
DEBUG_AGENT_BASE_TOOLS: tuple[str, ...] = (
    "parse_failure_signal",
    "get_workspace_debug_context",
    "get_code_file_structure",
    "search_text",
    "search_bash",
    "fetch_file",
    "fetch_files_batch",
    "run_validation",
    "record_hypothesis",
    "update_hypothesis_status",
    "append_hypothesis_evidence",
    "list_hypotheses",
    "read_todos",
    "add_todo",
    "update_todo_status",
    "get_available_tasks",
)

# DAP debugger tools (A3) — only included when ctx.local_mode is True, since
# they require a running VS Code / DAP debug adapter.
DEBUG_AGENT_DAP_TOOLS: tuple[str, ...] = (
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
)

# Potpie terminal tools (VS Code extension tunnel) — local_mode only.
# Used for compile/run, launch.json creation/cleanup, and other shell tasks
# that must not go through sandbox_shell (read-only extension mode).
DEBUG_AGENT_TERMINAL_TOOLS: tuple[str, ...] = (
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
)

# Set form for fast membership checks when filtering tools in non-local mode.
DAP_TOOL_NAMES: set[str] = set(DEBUG_AGENT_DAP_TOOLS)
TERMINAL_TOOL_NAMES: set[str] = set(DEBUG_AGENT_TERMINAL_TOOLS)
LOCAL_MODE_ONLY_TOOL_NAMES: set[str] = DAP_TOOL_NAMES | TERMINAL_TOOL_NAMES


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
        local_mode = ctx.local_mode if ctx else False
        logger.info(
            "[DEBUG DebugAgent] _build_agent called: local_mode=%s user_id=%s conversation_id=%s",
            local_mode,
            getattr(ctx, "user_id", None),
            getattr(ctx, "conversation_id", None),
        )
        agent_config = AgentConfig(
            role="Debugging and Code Analysis Specialist",
            goal="Provide comprehensive debugging solutions and code analysis by identifying root causes, tracing code flows, and delivering precise fixes. For general queries, maintain a conversational approach while grounding responses in code context.",
            backstory="""
                    You are a seasoned debugging engineer with deep expertise in systematic problem-solving, root cause analysis, and code comprehension. You excel at:
                    1. Conversational code exploration and Q&A - helping users understand codebases naturally
                    2. Systematic debugging - when faced with bugs, you follow rigorous methodologies to find root causes
                    3. Strategic thinking - you fix problems at their source, not just patch symptoms
                    4. Code navigation - you expertly traverse code structures, file relationships, and call paths
                    5. Contextual understanding - you build comprehensive mental models of how code fits together

                    You adapt your approach: conversational for questions, methodical for debugging. You use focused todo tools (read_todos, add_todo, update_todo_status, get_available_tasks) to track progress and ensure thoroughness.
                """,
            tasks=[
                TaskConfig(
                    description=debug_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results. For debugging tasks, includes root cause analysis, fix location rationale, and implementation details.",
                )
            ],
        )

        tools = self.tools_provider.get_tools(
            list(DEBUG_AGENT_BASE_TOOLS)
            + list(DEBUG_AGENT_DAP_TOOLS)
            + list(DEBUG_AGENT_TERMINAL_TOOLS),
        )
        if not local_mode:
            tools = [
                tool
                for tool in tools
                if getattr(tool, "name", None) not in LOCAL_MODE_ONLY_TOOL_NAMES
            ]
            logger.info(
                "DebugAgent: local_mode=False - excluding DAP and terminal tools"
            )

        logger.info(
            "[DEBUG DebugAgent] tool names given to agent (local_mode=%s): %s",
            local_mode,
            sorted(getattr(t, "name", str(t)) for t in tools),
        )
        logger.info("Using PydanticDeepDebugAgent")
        return PydanticDeepDebugAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent(ctx).run_stream(ctx):
            yield chunk


# debug_task_prompt is imported from debug_agent_prompt above.
# It is defined there to keep the prompt importable without pulling in the full
# agent stack (provider_service → botocore, etc.), enabling lightweight smoke tests.
