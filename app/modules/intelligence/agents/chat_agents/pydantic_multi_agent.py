import functools
import re
import traceback
import json
from typing import List, AsyncGenerator, Sequence, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import anyio

from pydantic_ai import Agent, Tool, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ModelResponse,
    TextPart,
    ImageUrl,
    UserContent,
    ModelMessage,
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError
from pydantic_ai.usage import UsageLimits
from langchain_core.tools import StructuredTool

from .tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
    get_delegation_call_message,
    get_delegation_response_message,
    get_delegation_info_content,
    get_delegation_result_content,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .agent_config import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger

from ..chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)

logger = setup_logger(__name__)


class AgentType(Enum):
    """Types of specialized agents in the multi-agent system"""

    SUPERVISOR = "supervisor"
    THINK_EXECUTE = "think_execute"  # Generic Think and Execute Agent


@dataclass
class AgentDelegationResult:
    """Result from agent delegation"""

    agent_type: AgentType
    result: str
    success: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception:
            logger.exception("Exception in tool function", tool_name=tool_func.__name__)
            return "An internal error occurred. Please try again later."

    return wrapper


def is_delegation_tool(tool_name: str) -> bool:
    """Check if a tool call is a delegation to a subagent"""
    return tool_name.startswith("delegate_to_")


def extract_agent_type_from_delegation_tool(tool_name: str) -> str:
    """Extract agent type from delegation tool name"""
    if tool_name.startswith("delegate_to_"):
        return tool_name[12:]  # Remove "delegate_to_" prefix
    return tool_name


def extract_task_result_from_response(response: str) -> str:
    """
    Extract the Task Result section from a subagent response.
    Returns the full Task Result without truncation - can include detailed content and code snippets.
    If no Task Result section is found, return the full response.
    Enhanced to handle error cases and provide better fallbacks.
    """
    import re

    if not response or not response.strip():
        logger.warning("Empty response provided to extract_task_result_from_response")
        return ""

    # Check for error indicators first
    error_indicators = [
        r"(?i)❌\s*error",
        r"(?i)⚠️\s*error",
        r"(?i)🚨\s*error",
        r"(?i)error\s*occurred",
        r"(?i)failed\s*to",
        r"(?i)exception",
        r"(?i)traceback",
    ]

    for error_pattern in error_indicators:
        if re.search(error_pattern, response):
            logger.info("Error indicators found in response, returning full response")
            return response

    # Pattern to match Task Result section (case insensitive)
    # Updated patterns to better capture the end of result sections
    patterns = [
        r"(?i)#{1,4}\s*task\s*result[:\s]*\n(.*?)(?=\n#{1,4}\s*(?!task\s*result)\w+|\Z)",
        r"(?i)\*\*task\s*result[:\s]*\*\*\n(.*?)(?=\n\*\*(?!task\s*result)\w+|\Z)",
        r"(?i)task\s*result[:\s]*\n(.*?)(?=\n\w+:|\n#{1,4}\s*\w+|\Z)",
        r"(?i)## result[:\s]*\n(.*?)(?=\n#{1,4}\s*(?!result)\w+|\Z)",
        r"(?i)\*\*result[:\s]*\*\*\n(.*?)(?=\n\*\*(?!result)\w+|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary:
                logger.info(
                    f"Successfully extracted Task Result from subagent response (length: {len(summary)} chars)"
                )
                return summary

    # If no Task Result section is found, look for conclusion or final sections
    conclusion_patterns = [
        r"(?i)#{1,4}\s*conclusion[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
        r"(?i)#{1,4}\s*summary[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
        r"(?i)#{1,4}\s*findings[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary:
                logger.warning(
                    f"No Task Result found, but found conclusion/summary section (length: {len(summary)} chars)"
                )
                return summary

    # If still no structured section found, return the last meaningful paragraphs without truncation
    lines = response.strip().split("\n")
    if len(lines) > 10:  # If response is substantial, try to get the final content
        # Look for the last substantial paragraphs
        meaningful_lines = []
        for line in reversed(lines):
            line = line.strip()
            if line:  # Keep all non-empty lines including code blocks
                meaningful_lines.append(line)
                if len(meaningful_lines) >= 10:  # Get more content for detailed summary
                    break

        if meaningful_lines:
            summary = "\n".join(reversed(meaningful_lines))
            logger.warning(
                f"No Task Summary section found, using last meaningful content as detailed summary (length: {len(summary)} chars)"
            )
            return summary

    # Final fallback: return full response without truncation
    logger.warning(
        f"No Task Summary section found, returning full response (length: {len(response)} chars)"
    )
    return response


def create_tool_call_response(event: FunctionToolCallEvent) -> ToolCallResponse:
    """Create appropriate tool call response for regular or delegation tools"""
    tool_name = event.part.tool_name

    # Safely parse tool arguments with error handling for malformed JSON
    try:
        args_dict = event.part.args_as_dict()
    except (ValueError, json.JSONDecodeError) as json_error:
        # Handle incomplete/malformed JSON in tool arguments
        # This can happen when the model generates truncated JSON, often due to:
        # - Token limits during streaming
        # - Model stopping mid-generation
        # - Network issues interrupting the response
        raw_args = getattr(event.part, "args", "N/A")
        logger.error(
            f"JSON parsing error in tool call '{tool_name}': {json_error}. "
            f"Tool args (raw, first 300 chars): {str(raw_args)[:300]}. "
            f"This may cause issues when pydantic_ai tries to serialize the message history."
        )
        # Return empty args dict to allow processing to continue
        # Note: This won't prevent errors when pydantic_ai tries to serialize
        # the message history later, but it allows the current tool call to proceed
        args_dict = {}

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        task_description = args_dict.get("task_description", "")
        context = args_dict.get("context", "")

        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_CALL,
            tool_name=tool_name,
            tool_response=get_delegation_call_message(agent_type),
            tool_call_details={
                "summary": get_delegation_info_content(
                    agent_type, task_description, context
                )
            },
        )
    else:
        return ToolCallResponse(
            call_id=event.part.tool_call_id or "",
            event_type=ToolCallEventType.CALL,
            tool_name=tool_name,
            tool_response=get_tool_run_message(tool_name),
            tool_call_details={
                "summary": get_tool_call_info_content(tool_name, args_dict)
            },
        )


def create_tool_result_response(event: FunctionToolResultEvent) -> ToolCallResponse:
    """Create appropriate tool result response for regular or delegation tools"""
    tool_name = event.result.tool_name or "unknown tool"

    if is_delegation_tool(tool_name):
        agent_type = extract_agent_type_from_delegation_tool(tool_name)
        result_content = str(event.result.content) if event.result.content else ""

        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.DELEGATION_RESULT,
            tool_name=tool_name,
            tool_response=get_delegation_response_message(agent_type),
            tool_call_details={
                "summary": get_delegation_result_content(agent_type, result_content)
            },
        )
    else:
        return ToolCallResponse(
            call_id=event.result.tool_call_id or "",
            event_type=ToolCallEventType.RESULT,
            tool_name=tool_name,
            tool_response=get_tool_response_message(tool_name),
            tool_call_details={
                "summary": get_tool_result_info_content(tool_name, event.result.content)
            },
        )


class PydanticMultiAgent(ChatAgent):
    """
    Multi-agent system using Pydantic AI with agent delegation patterns.

    This system consists of:
    1. A supervisor agent that coordinates tasks
    2. Specialized delegate agents for specific tasks
    3. Context passing between agents
    4. Usage tracking across all agents
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None = None,
        delegate_agents: Optional[Dict[AgentType, AgentConfig]] = None,
    ):
        """Initialize the multi-agent system with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter
        self.llm_provider = llm_provider
        self.tools = tools
        self.config = config
        self.mcp_servers = mcp_servers or []

        # Clean tool names (no spaces for pydantic agents)
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Initialize delegate agents
        self.delegate_agents = delegate_agents or self._create_default_delegate_agents()

        # Store agent instances
        self._agent_instances: Dict[AgentType, Agent] = {}
        self._supervisor_agent: Optional[Agent] = None

    # Constants for agent instructions
    DELEGATE_AGENT_INSTRUCTIONS = """You are a focused task execution agent with access to all available tools. Execute the assigned task efficiently and provide clear, concise results.

**CODE MANAGEMENT:**
- Use code changes tools (add_file_to_changes, update_file_lines, insert_lines, delete_lines) instead of including code in response text
- Use show_updated_file and show_diff to display changes effectively
- Keep responses concise and avoid large code blocks in your text

**EXECUTION:**
- Execute tasks completely without asking for clarification unless absolutely critical
- Make reasonable assumptions and mention them
- Use tools to gather information and perform actions
- Return focused, actionable results"""

    # Tool name constants
    TOOL_NAME_SHOW_UPDATED_FILE = "show_updated_file"
    TOOL_NAME_SHOW_DIFF = "show_diff"

    @staticmethod
    def _create_error_response(message: str) -> ChatAgentResponse:
        """Create a standardized error response"""
        return ChatAgentResponse(
            response=f"\n\n{message}\n\n",
            tool_calls=[],
            citations=[],
        )

    @staticmethod
    async def _yield_text_stream_events(
        request_stream: Any, agent_type: str = "agent"
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Yield text streaming events from a request stream"""
        async for event in request_stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                yield ChatAgentResponse(
                    response=event.part.content,
                    tool_calls=[],
                    citations=[],
                )
            if isinstance(event, PartDeltaEvent) and isinstance(
                event.delta, TextPartDelta
            ):
                yield ChatAgentResponse(
                    response=event.delta.content_delta,
                    tool_calls=[],
                    citations=[],
                )

    @staticmethod
    def _handle_stream_error(
        error: Exception, context: str = "model request stream"
    ) -> Optional[ChatAgentResponse]:
        """Handle streaming errors and return appropriate response"""
        if isinstance(error, (ModelRetry, AgentRunError, UserError)):
            logger.warning(f"Pydantic-ai error in {context}: {error}")
            return PydanticMultiAgent._create_error_response(
                "*Encountered an issue while processing your request. Trying to recover...*"
            )
        elif isinstance(error, anyio.WouldBlock):
            logger.warning(f"{context} would block - continuing...")
            return None  # Signal to continue
        elif isinstance(error, ValueError):
            error_str = str(error)
            if (
                "json" in error_str.lower()
                or "parse" in error_str.lower()
                or "EOF" in error_str
            ):
                logger.error(
                    f"JSON parsing error in {context} (likely from malformed tool call in message history): {error}. "
                    f"This may indicate a truncated or incomplete tool call from a previous iteration. "
                    f"Full traceback:\n{traceback.format_exc()}"
                )
                return PydanticMultiAgent._create_error_response(
                    "*Encountered a parsing error. Skipping this step and continuing...*"
                )
            else:
                raise  # Re-raise if it's a different ValueError
        else:
            logger.exception("Unexpected error", context=context)
            if "json" in str(error).lower() or "parse" in str(error).lower():
                return PydanticMultiAgent._create_error_response(
                    "*Encountered a parsing error. Skipping this step and continuing...*"
                )
            return PydanticMultiAgent._create_error_response(
                "*An unexpected error occurred. Continuing...*"
            )

    @staticmethod
    def _create_delegation_prompt(
        task_description: str, project_context: str, task_specific_context: str = ""
    ) -> str:
        """Create the delegation prompt for subagents"""
        full_context_parts = [project_context]
        if task_specific_context and task_specific_context.strip():
            full_context_parts.append(f"Task-specific context: {task_specific_context}")

        full_context = "\n\n".join(full_context_parts)

        return f"""Execute this focused task for the supervisor:

**TASK:**
{task_description}

**PROJECT CONTEXT:**
{full_context}

**YOUR MISSION:**
Execute the task above and return ONLY the specific, actionable result the supervisor needs.

**OUTPUT FORMAT:**
Start your response with "## Task Result" and then provide the focused answer.

**CRITICAL GUIDELINES:**
1. Use tools to gather information, analyze code, or perform actions as needed
2. Be specific and concise - avoid broad explanations or context gathering
3. Focus on answering the exact question or completing the exact task
4. If you make code changes, use show_updated_file and show_diff to display them
5. Don't restate the problem - just solve it and report the result

**RESULT:** Should be specific, actionable, and immediately usable by the supervisor."""

    @staticmethod
    def _format_delegation_error(
        agent_type: AgentType,
        task_description: str,
        error_type: str,
        error_message: str,
        raw_response: str = "",
    ) -> str:
        """Format error responses for delegation failures"""
        if error_type == "no_result":
            return f"""
## Task Result

❌ **ERROR: No valid task result found**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Issue:** The subagent did not provide a properly formatted Task Result section

**Raw Response (truncated):**
{raw_response[:500]}{'...' if len(raw_response) > 500 else ''}

**Recommendation:** The supervisor should retry the delegation with clearer instructions.
            """.strip()
        elif error_type == "empty_result":
            return f"""
## Task Result

❌ **ERROR: Empty or invalid result**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Issue:** The subagent returned no output or an invalid response

**Recommendation:** The supervisor should retry the delegation or try a different approach.
            """.strip()
        else:  # exception
            return f"""
## Task Result

❌ **ERROR: Delegation failed**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Error:** {error_message}
**Error Type:** {error_type}

**Recommendation:** The supervisor should investigate the error and retry with a different approach or agent.
            """.strip()

    @staticmethod
    async def _collect_agent_streaming_response(
        agent: Agent, user_prompt: str, agent_type: str = "agent"
    ) -> str:
        """Collect streaming response from an agent run"""
        full_response = ""

        async with agent.iter(
            user_prompt=user_prompt,
            usage_limits=UsageLimits(request_limit=None),
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    # Stream tokens from the model's request
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent) and isinstance(
                                event.part, TextPart
                            ):
                                full_response += event.part.content
                                logger.debug(
                                    f"[{agent_type}] Streaming chunk: {event.part.content[:100]}{'...' if len(event.part.content) > 100 else ''}"
                                )
                            if isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                full_response += event.delta.content_delta
                                logger.debug(
                                    f"[{agent_type}] Streaming delta: {event.delta.content_delta[:100]}{'...' if len(event.delta.content_delta) > 100 else ''}"
                                )
                elif Agent.is_end_node(node):
                    break

        return full_response

    @staticmethod
    async def _yield_tool_result_event(
        event: FunctionToolResultEvent,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Yield appropriate response for tool result events"""
        tool_name = event.result.tool_name or "unknown"
        tool_result = create_tool_result_response(event)

        # For show_updated_file and show_diff, append content directly to response
        # instead of going through tool_result_info - these stream directly to user
        if tool_name in (
            PydanticMultiAgent.TOOL_NAME_SHOW_UPDATED_FILE,
            PydanticMultiAgent.TOOL_NAME_SHOW_DIFF,
        ):
            content = str(event.result.content) if event.result.content else ""
            yield ChatAgentResponse(
                response=content,
                tool_calls=[tool_result],
                citations=[],
            )
        else:
            yield ChatAgentResponse(
                response="",
                tool_calls=[tool_result],
                citations=[],
            )

    async def _process_agent_run_nodes(
        self, run: Any, context: str = "agent"
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process nodes from an agent run and yield responses"""
        async for node in run:
            if Agent.is_model_request_node(node):
                # Stream tokens from the model's request
                try:
                    async with node.stream(run.ctx) as request_stream:
                        async for chunk in self._yield_text_stream_events(
                            request_stream, context
                        ):
                            yield chunk
                except Exception as e:
                    error_response = self._handle_stream_error(
                        e, f"{context} model request stream"
                    )
                    if error_response:
                        yield error_response
                    continue

            elif Agent.is_call_tools_node(node):
                # Handle tool calls and results
                async for response in self._process_tool_call_node(node, run.ctx):
                    yield response

            elif Agent.is_end_node(node):
                logger.info(f"{context} result streamed successfully!!")
                break

    async def _process_tool_call_node(
        self, node: Any, ctx: Any
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process tool call nodes and yield responses"""
        try:
            async with node.stream(ctx) as handle_stream:
                async for event in handle_stream:
                    if isinstance(event, FunctionToolCallEvent):
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[create_tool_call_response(event)],
                            citations=[],
                        )
                    if isinstance(event, FunctionToolResultEvent):
                        async for response in self._yield_tool_result_event(event):
                            yield response
        except (
            ModelRetry,
            AgentRunError,
            UserError,
        ) as pydantic_error:
            logger.warning(f"Pydantic-ai error in tool call stream: {pydantic_error}")
            yield self._create_error_response(
                "*Encountered an issue while calling tools. Trying to recover...*"
            )
        except anyio.WouldBlock:
            logger.warning("Tool call stream would block - continuing...")
        except Exception:
            logger.exception("Unexpected error in tool call stream")
            yield self._create_error_response(
                "*An unexpected error occurred during tool execution. Continuing...*"
            )

    def _create_mcp_servers(self) -> List[MCPServerStreamableHTTP]:
        """Create MCP server instances from configuration"""
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"], timeout=10.0
                )
                mcp_toolsets.append(mcp_server_instance)
                logger.info(
                    f"Successfully created MCP server: {mcp_server.get('name', 'unknown')}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create MCP server {mcp_server.get('name', 'unknown')}: {e}"
                )
                continue
        return mcp_toolsets

    def _wrap_structured_tools(self, tools: Sequence[Any]) -> List[Tool]:
        """Convert tool instances (StructuredTool or similar) to PydanticAI Tool instances"""
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                function=handle_exception(tool.func),  # type: ignore
            )
            for tool in tools
        ]

    def _build_delegate_agent_tools(self) -> List[Tool]:
        """Build the tool list for delegate agents"""
        return self._wrap_structured_tools(self.tools)

    def _build_supervisor_agent_tools(self) -> List[Tool]:
        """Build the tool list for supervisor agent including delegation, todo, and code changes tools"""
        # Import tools here to avoid circular imports
        from app.modules.intelligence.tools.todo_management_tool import (
            create_todo_management_tools,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            create_code_changes_management_tools,
        )

        todo_tools = create_todo_management_tools()
        code_changes_tools = create_code_changes_management_tools()

        # Create delegation tools
        delegation_tools = []
        for agent_type in self.delegate_agents.keys():
            if agent_type == AgentType.THINK_EXECUTE:
                description = "🔨 DELEGATE TO TASK EXECUTION AGENT - Delegate focused work to save your context! Great for: information gathering, code searches, targeted implementations, debugging specific issues, analyzing code sections. The subagent will return clean, focused results. Use this liberally to break down complex tasks! IMPORTANT: Use the 'context' parameter to pass already-fetched information (file paths, code snippets, analysis results) to avoid redundant work - this is critical for efficiency!"
            else:
                description = f"🤖 DELEGATE TO {agent_type.value.upper()} - Delegate focused work to save your context! The subagent will return clean, focused results. Use this liberally to break down complex tasks! IMPORTANT: Use the 'context' parameter to pass already-fetched information (file paths, code snippets, analysis results) to avoid redundant work - this is critical for efficiency!"

            delegation_tools.append(
                Tool(
                    name=f"delegate_to_{agent_type.value}",
                    description=description,
                    function=self._create_delegation_function(agent_type),
                )
            )

        return (
            self._wrap_structured_tools(self.tools)
            + delegation_tools
            + self._wrap_structured_tools(todo_tools)
            + self._wrap_structured_tools(code_changes_tools)
        )

    def _create_default_delegate_agents(self) -> Dict[AgentType, AgentConfig]:
        """Create default specialized agents if none provided"""
        return {
            AgentType.THINK_EXECUTE: AgentConfig(
                role="Task Execution Specialist",
                goal="Execute specific tasks with clear, actionable results",
                backstory="""You are a focused task executor for specific, well-defined tasks.

You receive tasks that have clear, specific expected outputs. Your job is to provide exactly what the supervisor needs - no more, no less.

CRITICAL: Do ALL your work inside the "## Task Result" section.

Your approach:
1. Understand exactly what specific information is needed
2. Start the "## Task Result" section immediately
3. Provide the specific answer/information requested
4. Keep results concise and focused
5. Include only what the supervisor needs to make a decision

You are used for specific lookups, small implementations, and focused tasks - not broad analysis or context gathering.""",
                tasks=[
                    TaskConfig(
                        description="""Execute the specific, well-defined task assigned by the supervisor.

This task has a clear expected output. Provide exactly what the supervisor needs.

CRITICAL INSTRUCTIONS:
- Do ALL your work inside the "## Task Result" section
- Provide the specific information/answer requested
- Keep results concise and focused
- Don't provide broad analysis or context gathering
- Be efficient - supervisor needs specific information

TASK RESULT SECTION REQUIREMENTS:
- Start with "## Task Result" immediately
- Provide the specific answer/information requested
- Include any code, files, or data needed
- Keep it focused on what the supervisor asked for
- End with the specific result requested

Remember: You are used for specific lookups and focused tasks, not broad analysis.""",
                        expected_output="Specific task completion with concrete execution results and deliverables",
                    )
                ],
                max_iter=20,
            ),
        }

    def _create_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create a generic delegate agent with all tools for focused task execution"""
        if agent_type in self._agent_instances:
            return self._agent_instances[agent_type]

        agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=self._build_delegate_agent_tools(),
            mcp_servers=self._create_mcp_servers(),
            # Delegate agents get minimal instructions - the full task context comes from delegation
            instructions=self.DELEGATE_AGENT_INSTRUCTIONS,
            # result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            instrument=True,
        )
        self._agent_instances[agent_type] = agent
        return agent

    def _create_supervisor_agent(self, ctx: ChatContext) -> Agent:
        """Create the supervisor agent that coordinates other agents"""
        if self._supervisor_agent:
            return self._supervisor_agent

        # Prepare multimodal instructions if images are present
        multimodal_instructions = self._prepare_multimodal_instructions(ctx)

        supervisor_agent = Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=self._build_supervisor_agent_tools(),
            mcp_servers=self._create_mcp_servers(),
            instructions=f"""
            You are a problem-solving supervisor who orchestrates subagents to efficiently solve complex tasks.

            **🚀 MANDATORY PLANNING PHASE (DO THIS FIRST):**
            1. **Analyze:** Understand the request, identify objectives, dependencies, and constraints
            2. **Break down:** Split into logical, delegable chunks (self-contained, clear outcomes, minimal interdependencies)
            3. **Create TODOs:** Use `create_todo` for every step (main tasks → subtasks), mark dependencies, set status to "pending"
            4. **Plan delegation:** Identify what to delegate, determine execution order, plan context needs
            5. **Document:** Summarize problem, list chunks, explain strategy, note assumptions

            **📋 EXECUTION & ADAPTATION:**
            - Execute systematically: Follow your plan, delegate tasks with proper context
            - Track progress: Update todo status (pending → in_progress → completed), add notes as you learn
            - Adapt: Update plan and TODOs based on discoveries - your plan can evolve!
            - Verify: Ensure all TODOs complete and objective met

            **🎯 INTELLIGENT DELEGATION:**
            **Why delegate:** Subagents return clean results via "## Task Result" format, saving your context.

            **When to delegate:** targeted analysis, research, implementation slices etc.
            for information gathering pertaining to the problem don't delegate task since the gathered context might be useful later on aswell

            **How to delegate effectively:**
            - Break tasks into focused chunks with clear success criteria
            - **CRITICAL:** Pass already-fetched context via `context` parameter (file paths, code snippets, analysis results, config values)
            - Group related tasks to share context efficiently
            - Example context: "Main function in app/main.py:45-67 uses Config class from app/core/config.py with 'database_url' property. Error at line 52."

            **Code Management:**
            - **CRITICAL:** All your code changes for this session are tracked in the code changes manager - it persists throughout the conversation
            - **ALWAYS use code changes tools** (not response text): `add_file_to_changes`, `update_file_lines`, `replace_in_file`, `insert_lines`, `delete_lines`
            - **For precise editing, ALWAYS fetch files with line numbers:** Use `fetch_file` with `with_line_numbers=true` to see exact line numbers and indentation before editing. This ensures you know the exact line numbers and indentation to use with `insert_lines`, `delete_lines`, and `update_file_lines`
            - **CRITICAL: Preserve proper indentation:** When using `insert_lines` or `update_file_lines`, match the indentation of surrounding lines exactly. Fetch the file first to see the exact indentation pattern, then preserve it in your updates
            - **ALWAYS verify your edits:** After using `insert_lines` or `update_file_lines`, fetch the updated lines in context (with surrounding lines) to verify:
              * Indentation is correct and matches surrounding code
              * Content was inserted/updated as expected
              * Code structure is intact
              * If verification fails, fix it immediately using the appropriate tool
            - **Precise line operations:** Use `insert_lines` to add code at specific line numbers, `delete_lines` to remove specific line ranges, and `update_file_lines` to replace specific lines
            - **Check your progress:** Use `get_session_metadata` to see all files you've modified, timestamps, descriptions, and line counts
            - **Review changes:** Use `get_file_from_changes` to see file metadata, or `get_file_diff` (with project_id) to see diff against repository branch
            - **Before making changes:** Check `list_files_in_changes` or `get_session_metadata` to see what's already been modified
            - Prefer targeted updates over full rewrites - use line numbers for precision
            - Display changes with BOTH `show_updated_file` (complete files) AND `show_diff` (change details, with project_id for repository diffs)
            - Why: Code in tools saves 70-85% tokens vs response text that accumulates in history
            - Write code only once, don't show changes and then update it in the code changes manager

            **🚀 PROACTIVE PROBLEM SOLVING:**
            - Solve completely without asking unless critical info is missing
            - Make reasonable assumptions, state them explicitly
            - Choose best approach when multiple options exist
            - Add steps to TODO and execute systematically

            Your Identity:
            Role: {self.config.role}
            Goal: {self.config.goal}

            {multimodal_instructions}

            CONTEXT: {self._create_supervisor_task_description(ctx)}
            """,
            # result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            # model_settings={"max_tokens": 64000},
            instrument=True,
        )
        self._supervisor_agent = supervisor_agent
        return supervisor_agent

    def _create_delegation_function(self, agent_type: AgentType):
        """Create a delegation function for a specific agent type"""

        async def delegate_function(
            ctx: RunContext[None], task_description: str, context: str = ""
        ) -> str:
            """Delegate a task to a specialized agent.

            Args:
                task_description: The specific task to delegate to the subagent. Should be clear and focused.
                context: Optional context to pass to the subagent (file paths, code snippets, analysis results,
                        configuration values, or any relevant information already fetched). This helps avoid
                        redundant work - pass any context the subagent would otherwise need to fetch.

            Returns:
                The task result from the subagent in "## Task Result" format.
            """
            try:
                logger.info(
                    f"Delegating task to {agent_type.value} agent: {task_description}"
                )

                # Create the delegate agent
                delegate_agent = self._create_agent(agent_type, self._current_context)

                # Build comprehensive context for the subagent
                project_context = self._create_project_context_info(
                    self._current_context
                )

                # Create delegation prompt using helper
                full_task = self._create_delegation_prompt(
                    task_description, project_context, context
                )

                # Run the delegate agent with independent usage tracking and streaming
                logger.info(f"Starting {agent_type.value} agent execution...")

                # Collect streaming response using helper
                full_response = await self._collect_agent_streaming_response(
                    delegate_agent, full_task, agent_type.value
                )

                logger.info(
                    f"Task completed by {agent_type.value} agent. Full response length: {len(full_response)} chars"
                )
                logger.info(
                    f"[{agent_type.value}] Full response: {full_response[:10000]}{'...' if len(full_response) > 10000 else ''}"
                )

                # Enhanced output parsing with error handling
                if full_response and len(full_response.strip()) > 0:
                    # Extract the Task Result section for clean supervisor context
                    summary = extract_task_result_from_response(full_response)

                    if summary and len(summary.strip()) > 0:
                        logger.info(
                            f"Extracted task result for supervisor (length: {len(summary)} chars)"
                        )
                        return summary
                    else:
                        # No valid result found, return formatted error
                        logger.warning(
                            f"No valid task result found in {agent_type.value} response"
                        )
                        return self._format_delegation_error(
                            agent_type,
                            task_description,
                            "no_result",
                            "No valid task result found",
                            full_response,
                        )
                else:
                    # No output or empty result
                    logger.error(
                        f"Empty or invalid result from {agent_type.value} agent"
                    )
                    return self._format_delegation_error(
                        agent_type,
                        task_description,
                        "empty_result",
                        "Empty or invalid result",
                        full_response,
                    )

            except Exception as e:
                logger.exception(
                    "Error in delegation",
                    agent_type=agent_type.value,
                    task_description=task_description,
                )
                return self._format_delegation_error(
                    agent_type, task_description, type(e).__name__, str(e), ""
                )

        return delegate_function

    def _prepare_multimodal_instructions(self, ctx: ChatContext) -> str:
        """Prepare multimodal-specific instructions when images are present"""
        if not ctx.has_images():
            return ""

        all_images = ctx.get_all_images()
        current_images = ctx.get_current_images_only()
        context_images = ctx.get_context_images_only()

        return f"""
        MULTIMODAL ANALYSIS INSTRUCTIONS:
        You have access to {len(all_images)} image(s) - {len(current_images)} from the current message and {len(context_images)} from conversation history.

        CRITICAL GUIDELINES FOR ACCURATE ANALYSIS:
        1. **ONLY analyze what you can clearly see** - Do not infer or guess about unclear details
        2. **Distinguish between current and historical images** - Focus primarily on current message images
        3. **State uncertainty** - If you cannot clearly see something, say "I cannot clearly see..." instead of guessing
        4. **Be specific** - Reference exact text, colors, shapes, or elements you observe
        5. **Avoid assumptions** - Do not assume context beyond what's explicitly visible

        ANALYSIS APPROACH:
        - **Current Images**: These are directly related to the user's current query
        - **Historical Images**: These provide context but may not be directly relevant
        - **Text Recognition**: Only transcribe text that is clearly legible
        - **UI Elements**: Only describe elements that are clearly visible and identifiable
        - **Error Messages**: Only report errors that are clearly displayed and readable

        IMPORTANT: If an image is unclear, blurry, or doesn't contain the type of content the user is asking about, explicitly state this rather than making assumptions.
        """

    def _create_task_description(
        self,
        task_config: TaskConfig,
        ctx: ChatContext,
        agent_type: AgentType,
    ) -> str:
        """Create a task description from task configuration for delegate agents"""
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        # Add image context information
        image_context = ""
        if ctx.has_images():
            all_images = ctx.get_all_images()
            image_details = []
            for attachment_id, image_data in all_images.items():
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                image_details.append(f"- {file_name} ({file_size} bytes)")

            image_context = f"""
            ATTACHED IMAGES:
            {chr(10).join(image_details)}

            Image Analysis Notes:
            - These images are provided for visual analysis and debugging
            - Reference specific details from the images in your response
            - Correlate visual evidence with the user's query
            """

        return f"""
                CONTEXT:
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}
                Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

                {image_context}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                TASK HANDLING: (follow the method below if the user asks you to execute your task for your role and goal)
                {task_config.description}

                INSTRUCTIONS:
                1. Gather the necessary information
                2. {"Analyze the provided images in detail and " if ctx.has_images() else ""}Process and synthesize the gathered information
                3. Format your response in markdown unless explicitly asked to output in a different format, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. {"Reference specific details from the images when relevant" if ctx.has_images() else "Provide clear explanations"}
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                - You have access to tools in MCP Servers too, use them effectively. These mcp servers provide you with tools user might ask you to perform tasks on
                {"- Provide detailed image analysis when images are present" if ctx.has_images() else ""}
            """

    def _create_project_context_info(self, ctx: ChatContext) -> str:
        """Create project context information for both supervisor and subagents"""
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        # Add image context information
        image_context = ""
        if ctx.has_images():
            all_images = ctx.get_all_images()
            image_details = []
            for attachment_id, image_data in all_images.items():
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                image_details.append(f"- {file_name} ({file_size} bytes)")

            image_context = f"""
ATTACHED IMAGES:
{chr(10).join(image_details)}

Image Analysis Notes:
- These images are provided for visual analysis and debugging
- Reference specific details from the images in your response
- Correlate visual evidence with the user's query"""

        context_parts = []

        # Project information
        if ctx.project_id:
            context_parts.append(f"Project: {ctx.project_name} (ID: {ctx.project_id})")

        # Node information
        if ctx.node_ids:
            context_parts.append(f"Nodes: {', '.join(ctx.node_ids)}")

        # Image context
        if image_context:
            context_parts.append(image_context.strip())

        # Additional context
        if ctx.additional_context:
            context_parts.append(f"Additional Context: {ctx.additional_context}")

        return "\n".join(context_parts) if context_parts else "No additional context"

    def _create_supervisor_task_description(self, ctx: ChatContext) -> str:
        """Create a task description for the supervisor agent"""
        return self._create_project_context_info(ctx)

    def _create_multimodal_user_content(
        self, ctx: ChatContext
    ) -> Sequence[UserContent]:
        """Create multimodal user content with images using PydanticAI's ImageUrl"""
        content: List[UserContent] = [ctx.query]

        # Add current images to the content
        current_images = ctx.get_current_images_only()
        logger.info(
            f"Processing {len(current_images)} current images for multimodal content"
        )

        for attachment_id, image_data in current_images.items():
            try:
                # Validate image data structure
                if not isinstance(image_data, dict):
                    logger.error(
                        f"Invalid image data structure for {attachment_id}: {type(image_data)}"
                    )
                    continue

                # Validate required fields
                if "base64" not in image_data:
                    logger.error(f"Missing base64 data for image {attachment_id}")
                    continue

                base64_data = image_data["base64"]
                if not isinstance(base64_data, str) or not base64_data:
                    logger.error(f"Invalid base64 data for image {attachment_id}")
                    continue

                # Validate base64 format
                try:
                    import base64

                    base64.b64decode(base64_data)
                except Exception:
                    logger.exception(
                        "Invalid base64 format for image",
                        attachment_id=attachment_id,
                    )
                    continue

                # Get mime type with better fallback
                mime_type = image_data.get("mime_type", "image/jpeg")
                if (
                    not mime_type
                    or not isinstance(mime_type, str)
                    or not mime_type.startswith("image/")
                ):
                    logger.warning(
                        f"Invalid mime type for image {attachment_id}: {mime_type}, defaulting to image/jpeg"
                    )
                    mime_type = "image/jpeg"

                # Create data URL
                data_url = f"data:{mime_type};base64,{base64_data}"

                # Log image details for debugging
                file_name = image_data.get("file_name", "unknown")
                file_size = image_data.get("file_size", 0)
                logger.info(
                    f"Adding image {attachment_id} ({file_name}, {file_size} bytes, {mime_type}) to multimodal content"
                )

                content.append(ImageUrl(url=data_url))
                logger.info(
                    f"Successfully added image {attachment_id} to multimodal content"
                )

            except Exception:
                logger.exception(
                    "Failed to add image to content",
                    attachment_id=attachment_id,
                )
                continue

        # If no current images, add context images as fallback
        if not current_images:
            context_images = ctx.get_context_images_only()
            logger.info(
                f"No current images found, processing {len(context_images)} context images as fallback"
            )

            for attachment_id, image_data in context_images.items():
                try:
                    # Apply same validation as above
                    if not isinstance(image_data, dict) or "base64" not in image_data:
                        logger.error(f"Invalid context image data for {attachment_id}")
                        continue

                    base64_data = image_data["base64"]
                    if not isinstance(base64_data, str) or not base64_data:
                        logger.error(
                            f"Invalid base64 data for context image {attachment_id}"
                        )
                        continue

                    # Validate base64 format
                    try:
                        import base64

                        base64.b64decode(base64_data)
                    except Exception:
                        logger.exception(
                            "Invalid base64 format for context image",
                            attachment_id=attachment_id,
                        )
                        continue

                    mime_type = image_data.get("mime_type", "image/jpeg")
                    if (
                        not mime_type
                        or not isinstance(mime_type, str)
                        or not mime_type.startswith("image/")
                    ):
                        mime_type = "image/jpeg"

                    data_url = f"data:{mime_type};base64,{base64_data}"

                    file_name = image_data.get("file_name", "unknown")
                    file_size = image_data.get("file_size", 0)
                    logger.info(
                        f"Adding context image {attachment_id} ({file_name}, {file_size} bytes, {mime_type}) to multimodal content"
                    )

                    content.append(ImageUrl(url=data_url))
                    logger.info(
                        f"Successfully added context image {attachment_id} to multimodal content"
                    )
                except Exception:
                    logger.exception(
                        "Failed to add context image to content",
                        attachment_id=attachment_id,
                    )
                    continue

        logger.info(
            f"Final multimodal content has {len(content)} elements: 1 text + {len(content)-1} images"
        )
        return content

    async def _prepare_multimodal_message_history(
        self, ctx: ChatContext
    ) -> List[ModelMessage]:
        """Prepare message history with multimodal support"""
        history_messages = []

        for msg in ctx.history:
            # For now, keep history as text-only to avoid token bloat
            # Images are only added to the current query
            history_messages.append(ModelResponse([TextPart(content=str(msg))]))

        return history_messages

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow with multi-agent coordination"""
        logger.info(
            f"Running pydantic multi-agent system {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Reset todo manager for this agent run to ensure isolation
        from app.modules.intelligence.tools.todo_management_tool import (
            _reset_todo_manager,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            _reset_code_changes_manager,
        )

        _reset_todo_manager()
        _reset_code_changes_manager()
        logger.info("🔄 Reset todo manager and code changes manager for new agent run")

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent"
            )
            return await self._run_multimodal(ctx)
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only"
                )
            # Use standard PydanticAI multi-agent for text-only
            return await self._run_standard(ctx)

    async def _run_standard(self, ctx: ChatContext) -> ChatAgentResponse:
        """Standard text-only multi-agent execution"""
        try:
            # Prepare message history
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            # Try to initialize MCP servers with timeout handling
            try:
                async with supervisor_agent.run_mcp_servers():
                    resp = await supervisor_agent.run(
                        user_prompt=ctx.query,
                        message_history=message_history,
                    )
            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                error_detail = f"{type(mcp_error).__name__}: {str(mcp_error)}"
                logger.warning(
                    f"MCP server initialization failed in standard run: {error_detail}",
                    exc_info=True,
                )
                # Check if it's a JSON parsing error
                if (
                    "json" in str(mcp_error).lower()
                    or "parse" in str(mcp_error).lower()
                ):
                    logger.error(
                        f"JSON parsing error during MCP server initialization in standard run - MCP server may be returning malformed or incomplete JSON. Full traceback:\n{traceback.format_exc()}"
                    )
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                resp = await supervisor_agent.run(
                    user_prompt=ctx.query,
                    message_history=message_history,
                )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.exception("Error in standard multi-agent run method")
            return ChatAgentResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                tool_calls=[],
                citations=[],
            )

    async def _run_multimodal(self, ctx: ChatContext) -> ChatAgentResponse:
        """Multimodal multi-agent execution using PydanticAI's native multimodal capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            resp = await supervisor_agent.run(
                user_prompt=multimodal_content,
                message_history=message_history,
            )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception:
            logger.exception("Error in multimodal multi-agent run method")
            # Fallback to standard execution
            logger.info("Falling back to standard text-only execution")
            return await self._run_standard(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multi-agent response with delegation support"""
        logger.info(
            f"Running pydantic multi-agent stream {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent streaming"
            )
            async for chunk in self._run_multimodal_stream(ctx):
                yield chunk
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only streaming"
                )
            # Use standard PydanticAI multi-agent streaming for text-only
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_multimodal_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multimodal multi-agent response using PydanticAI's native capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create supervisor agent
            supervisor_agent = self._create_supervisor_agent(ctx)

            # Stream the response
            async with supervisor_agent.iter(
                user_prompt=multimodal_content,
                message_history=message_history,
                usage_limits=UsageLimits(
                    request_limit=None
                ),  # No request limit for long-running tasks
            ) as run:
                async for response in self._process_agent_run_nodes(
                    run, "multimodal multi-agent"
                ):
                    yield response

        except Exception:
            logger.exception("Error in multimodal multi-agent stream")
            # Fallback to standard streaming
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_standard_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Standard multi-agent streaming execution with MCP server support"""
        # Create supervisor agent directly
        supervisor_agent = self._create_supervisor_agent(ctx)

        try:
            # Try to initialize MCP servers with timeout handling
            try:
                async with supervisor_agent.run_mcp_servers():
                    async with supervisor_agent.iter(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                        usage_limits=UsageLimits(
                            request_limit=None
                        ),  # No request limit for long-running tasks
                    ) as run:
                        async for response in self._process_agent_run_nodes(
                            run, "multi-agent"
                        ):
                            yield response

            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                error_detail = f"{type(mcp_error).__name__}: {str(mcp_error)}"
                logger.warning(
                    f"MCP server initialization failed in stream: {error_detail}",
                    exc_info=True,
                )
                # Check if it's a JSON parsing error
                if (
                    "json" in str(mcp_error).lower()
                    or "parse" in str(mcp_error).lower()
                ):
                    logger.error(
                        f"JSON parsing error during MCP server initialization in stream - MCP server may be returning malformed or incomplete JSON. Full traceback:\n{traceback.format_exc()}"
                    )
                logger.info("Continuing without MCP servers...")

                # Fallback without MCP servers
                async with supervisor_agent.iter(
                    user_prompt=ctx.query,
                    message_history=[
                        ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                    ],
                    usage_limits=UsageLimits(
                        request_limit=None
                    ),  # No request limit for long-running tasks
                ) as run:
                    async for response in self._process_agent_run_nodes(
                        run, "multi-agent"
                    ):
                        yield response

        except Exception:
            logger.exception("Error in standard multi-agent stream")
            yield ChatAgentResponse(
                response="\n\n*An error occurred during multi-agent streaming*\n\n",
                tool_calls=[],
                citations=[],
            )
