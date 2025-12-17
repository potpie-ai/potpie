"""Agent factory for creating supervisor and delegate agents"""

from typing import List, Dict, Optional, Callable, Any
from pydantic_ai import Agent, Tool
from pydantic_ai.mcp import MCPServerStreamableHTTP
from langchain_core.tools import StructuredTool

from .utils.delegation_utils import AgentType
from .utils.tool_utils import wrap_structured_tools, deduplicate_tools_by_name
from .agent_instructions import (
    DELEGATE_AGENT_INSTRUCTIONS,
    get_supervisor_instructions,
    prepare_multimodal_instructions,
)
from .utils.context_utils import create_supervisor_task_description
from app.modules.intelligence.agents.chat_agent import ChatContext
from ..agent_config import AgentConfig, TaskConfig
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentFactory:
    """Factory for creating supervisor and delegate agents"""

    def __init__(
        self,
        llm_provider: ProviderService,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None,
        delegate_agents: Dict[AgentType, AgentConfig],
        history_processor: Any,
        create_delegation_function: Callable[[AgentType], Callable],
    ):
        """Initialize the agent factory"""
        self.llm_provider = llm_provider
        self.tools = tools
        self.mcp_servers = mcp_servers or []
        self.delegate_agents = delegate_agents
        self.history_processor = history_processor
        self.create_delegation_function = create_delegation_function

        # Clean tool names (no spaces for pydantic agents)
        import re

        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Cache for agent instances
        self._agent_instances: Dict[AgentType, Agent] = {}
        self._supervisor_agent: Optional[Agent] = None

    def create_mcp_servers(self) -> List[MCPServerStreamableHTTP]:
        """Create MCP server instances from configuration"""
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"], timeout=10.0
                )
                mcp_toolsets.append(mcp_server_instance)
            except Exception as e:
                logger.warning(
                    f"Failed to create MCP server {mcp_server.get('name', 'unknown')}: {e}"
                )
                continue
        return mcp_toolsets

    def build_delegate_agent_tools(self) -> List[Tool]:
        """Build the tool list for delegate agents - includes ALL supervisor tools EXCEPT delegation tools.

        This ensures subagents have full capability to execute tasks independently without
        needing to delegate further. The subagent gets the same tools as the supervisor
        (todo management, code changes, requirement verification, etc.) to work autonomously.
        """
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.todo_management_tool import (
            create_todo_management_tools,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            create_requirement_verification_tools,
        )

        todo_tools = create_todo_management_tools()
        code_changes_tools = create_code_changes_management_tools()
        requirement_tools = create_requirement_verification_tools()

        # Subagents get ALL tools EXCEPT delegation tools - they execute, don't delegate
        all_tools = (
            wrap_structured_tools(self.tools)
            + wrap_structured_tools(todo_tools)
            + wrap_structured_tools(code_changes_tools)
            + wrap_structured_tools(requirement_tools)
        )
        return deduplicate_tools_by_name(all_tools)

    def build_supervisor_agent_tools(self) -> List[Tool]:
        """Build the tool list for supervisor agent including delegation, todo, code changes, and requirement verification tools"""
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.todo_management_tool import (
            create_todo_management_tools,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            create_requirement_verification_tools,
        )

        todo_tools = create_todo_management_tools()
        code_changes_tools = create_code_changes_management_tools()
        requirement_tools = create_requirement_verification_tools()

        # Create delegation tools
        delegation_tools = []
        for agent_type in self.delegate_agents.keys():
            if agent_type == AgentType.THINK_EXECUTE:
                description = """🔨 DELEGATE TO SUBAGENT - Spin up a focused worker with ALL your tools to execute specific tasks or reason through problems.

**WHAT IS A SUBAGENT:**
- A subagent is an isolated execution context with access to ALL your tools (except delegation)
- It receives ONLY what you explicitly provide: task_description and context
- It does NOT inherit your conversation history - it starts fresh with just your input
- Its work streams back to the user in real-time, then you get a summary result

**WHY DELEGATE:**
1. **Context Efficiency**: Your context stays clean - subagent's tool calls don't bloat your history
2. **Token Savings**: Heavy tool usage happens in subagent's context, not yours
3. **Parallelization**: Spin up MULTIPLE subagents simultaneously for independent tasks
4. **Focus**: Subagents work on one specific task without distraction
5. **Reasoning Tool**: Use delegation to pause and think through complex problems in isolation

**WHEN TO DELEGATE:**
- ✅ **Reasoning & Thinking**: When you need to pause, recollect thoughts, and figure out a problem - delegate reasoning tasks to think through the situation
- ✅ Any task requiring multiple tool calls (searches, file reads, analysis)
- ✅ Code implementations in specific files/modules
- ✅ Debugging and investigation tasks
- ✅ Code analysis and understanding tasks
- ✅ Research and comparison tasks
- ✅ ANY task you want executed in isolation

**CRITICAL - CONTEXT PARAMETER:**
Since subagents DON'T get your history, YOU MUST provide comprehensive context:
- File paths and line numbers you've already identified
- Code snippets relevant to the task
- Analysis results or findings from your previous work
- Configuration values, error messages, or specific details
- Everything the subagent needs to execute WITHOUT re-fetching

**PARALLELIZATION:**
Call this tool MULTIPLE TIMES in parallel for independent tasks:
- E.g., "Analyze module A" and "Analyze module B" simultaneously
- Each subagent works independently with its own context
- Results stream back and you coordinate the final synthesis

**OUTPUT:**
- Subagent streams its work in real-time (visible to user)
- You receive a "## Task Result" summary for coordination
- Use the result to inform your next steps"""
            else:
                description = f"""🤖 DELEGATE TO {agent_type.value.upper()} SUBAGENT - Isolated execution with full tool access.

**WHAT IS A SUBAGENT:**
- An isolated execution context with ALL your tools (except delegation)
- Receives ONLY what you provide: task_description + context
- Does NOT inherit conversation history - starts fresh
- Streams work to user, returns summary to you

**WHY DELEGATE:**
- Context Efficiency: Keeps your context clean
- Token Savings: Heavy work happens in subagent context
- Parallelization: Run multiple subagents simultaneously
- Focus: Dedicated execution for specific tasks

**CRITICAL - CONTEXT PARAMETER:**
Subagents DON'T get your history. Provide comprehensive context:
- Relevant file paths, line numbers, code snippets
- Previous findings and analysis results
- Everything needed for autonomous execution

**PARALLELIZATION:** Call multiple times in parallel for independent tasks."""

            delegation_tools.append(
                Tool(
                    name=f"delegate_to_{agent_type.value}",
                    description=description,
                    function=self.create_delegation_function(agent_type),
                )
            )

        all_tools = (
            wrap_structured_tools(self.tools)
            + delegation_tools
            + wrap_structured_tools(todo_tools)
            + wrap_structured_tools(code_changes_tools)
            + wrap_structured_tools(requirement_tools)
        )
        # Deduplicate tools by name before returning
        return deduplicate_tools_by_name(all_tools)

    def create_delegate_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create a generic delegate agent with all tools for focused task execution"""
        if agent_type in self._agent_instances:
            return self._agent_instances[agent_type]

        agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=self.build_delegate_agent_tools(),
            mcp_servers=self.create_mcp_servers(),
            # Delegate agents get minimal instructions - the full task context comes from delegation
            instructions=DELEGATE_AGENT_INSTRUCTIONS,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            history_processors=[self.history_processor],
            instrument=True,
        )
        self._agent_instances[agent_type] = agent
        return agent

    def create_supervisor_agent(self, ctx: ChatContext, config: AgentConfig) -> Agent:
        """Create the supervisor agent that coordinates other agents"""
        if self._supervisor_agent:
            return self._supervisor_agent

        # Prepare multimodal instructions if images are present
        multimodal_instructions = prepare_multimodal_instructions(ctx)

        # Get supervisor task description
        supervisor_task_description = create_supervisor_task_description(ctx)

        # Generate supervisor instructions
        instructions = get_supervisor_instructions(
            config_role=config.role,
            config_goal=config.goal,
            task_description=config.tasks[0].description if config.tasks else "",
            multimodal_instructions=multimodal_instructions,
            supervisor_task_description=supervisor_task_description,
        )

        supervisor_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=self.build_supervisor_agent_tools(),
            mcp_servers=self.create_mcp_servers(),
            instrument=True,
            instructions=instructions,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            history_processors=[self.history_processor],
        )
        self._supervisor_agent = supervisor_agent
        return supervisor_agent


def create_default_delegate_agents() -> Dict[AgentType, AgentConfig]:
    """Create default specialized agents if none provided"""
    return {
        AgentType.THINK_EXECUTE: AgentConfig(
            role="Task Execution Specialist",
            goal="Execute specific tasks with clear, actionable results",
            backstory="""You are a focused task executor that works in isolated context. Execute specific tasks completely and return only the final result to keep the supervisor's context clean.""",
            tasks=[
                TaskConfig(
                    description="""Execute the specific task assigned by the supervisor. Do all work in your isolated context and return only the final result in "## Task Result" format.""",
                    expected_output="Specific task completion with concrete execution results and deliverables",
                )
            ],
            max_iter=20,
        ),
    }
