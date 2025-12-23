import functools
import re
import traceback
import json
import asyncio
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
    ModelRequest,
    TextPart,
    ImageUrl,
    UserContent,
    ModelMessage,
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError, ModelHTTPError
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
from .history_processor import create_history_processor
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
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
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
        # Attempt to sanitize the underlying message part so future serialization succeeds
        try:
            safe_args = json.dumps(args_dict)
            setattr(event.part, "args", safe_args)
        except Exception as sanitize_error:
            logger.warning(
                "Unable to sanitize malformed tool call arguments for '%s': %s",
                tool_name,
                sanitize_error,
            )

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
        # Track the current supervisor run to extract message history for subagents
        self._current_supervisor_run: Optional[Any] = None
        # Initialize history processor for token-aware context management
        # The processor stores compressed messages internally for reuse within the same execution
        self._history_processor = create_history_processor(llm_provider)
        # Track active streaming tasks for delegation tools (tool_call_id -> queue)
        self._active_delegation_streams: Dict[str, asyncio.Queue] = {}
        # Cache results from streaming to avoid duplicate execution (task_key -> result)
        self._delegation_result_cache: Dict[str, str] = {}
        # Store streamed content for each delegation (cache_key -> list of chunks)
        self._delegation_streamed_content: Dict[str, List[ChatAgentResponse]] = {}
        # Map tool_call_id to cache_key for retrieving streamed content
        self._delegation_cache_key_map: Dict[str, str] = {}

        # Reset todo manager, code changes manager, and requirement manager on initialization to ensure fresh state
        from app.modules.intelligence.tools.todo_management_tool import (
            _reset_todo_manager,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            _reset_code_changes_manager,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            _reset_requirement_manager,
        )

        _reset_todo_manager()
        _reset_code_changes_manager()
        _reset_requirement_manager()
        logger.info(
            "🔄 Reset todo manager, code changes manager, and requirement manager on agent initialization"
        )

    # Constants for agent instructions
    DELEGATE_AGENT_INSTRUCTIONS = """You are a focused task execution subagent. You receive a specific task with all necessary context from the supervisor agent. Execute the task completely and stream your work back.

**YOUR ROLE:**
- You are a SUBAGENT - you execute focused tasks delegated by the supervisor
- You receive ONLY the task description and context provided by the supervisor - NO additional conversation history
- You have access to ALL the same tools as the supervisor (except delegation) - use them freely
- Your responses stream back to the user in real-time, so be verbose about your progress
- You may be asked to reason through problems, analyze situations, or think through complex issues - treat these as execution tasks

**WHAT YOU RECEIVE:**
- A clear task description from the supervisor
- Relevant context (file paths, code snippets, previous findings, what the supervisor has learned, current problems/questions) the supervisor chose to include
- Project information (ID, name, etc.)
- You do NOT receive the full conversation history - only what the supervisor explicitly passes

**EXECUTION APPROACH:**
1. Read the task and context carefully - this is ALL the information you have
2. For reasoning tasks: Synthesize what's known, identify gaps, and suggest next steps
3. For execution tasks: Use tools to gather any additional information you need
4. Execute the task completely - don't ask for clarification unless absolutely critical
5. Make reasonable assumptions and state them explicitly
6. Stream your thinking and progress - the user sees your work in real-time

**TOOL USAGE:**
- You have ALL supervisor tools: code analysis, file fetching, bash commands, code changes management, etc.
- Use code changes tools (add_file_to_changes, update_file_lines, insert_lines, delete_lines) for modifications
- Use show_updated_file and show_diff to display changes to the user
- Use fetch_file with with_line_numbers=true for precise editing

**OUTPUT FORMAT:**
- Stream your work as you go - the user sees everything in real-time
- End with "## Task Result" section containing a concise summary
- The Task Result should be actionable and complete - the supervisor uses this for coordination

**REMEMBER:**
- You are isolated - use the tools to find what you need
- Your streaming output shows the user your progress
- Be thorough but focused on the specific task"""

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
        from app.modules.intelligence.tools.reasoning_manager import (
            _get_reasoning_manager,
        )

        reasoning_manager = _get_reasoning_manager()
        async for event in request_stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                # Accumulate TextPart content for reasoning dump
                reasoning_manager.append_content(event.part.content)
                yield ChatAgentResponse(
                    response=event.part.content,
                    tool_calls=[],
                    citations=[],
                )
            if isinstance(event, PartDeltaEvent) and isinstance(
                event.delta, TextPartDelta
            ):
                # Accumulate TextPartDelta content for reasoning dump
                reasoning_manager.append_content(event.delta.content_delta)
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
            error_detail = f"{type(error).__name__}: {str(error)}"
            logger.error(
                f"Unexpected error in {context}: {error_detail}", exc_info=True
            )
            if "json" in str(error).lower() or "parse" in str(error).lower():
                return PydanticMultiAgent._create_error_response(
                    "*Encountered a parsing error. Skipping this step and continuing...*"
                )
            return PydanticMultiAgent._create_error_response(
                "*An unexpected error occurred. Continuing...*"
            )

    @staticmethod
    def _create_delegation_prompt(
        task_description: str,
        project_context: str,
        supervisor_context: str = "",
    ) -> str:
        """Create the delegation prompt for subagents.

        Subagents are ISOLATED - they receive ONLY:
        1. The task description
        2. Project context (IDs, names)
        3. Context explicitly provided by the supervisor

        They do NOT receive conversation history or previous tool results.
        """
        context_sections = []

        if project_context:
            context_sections.append(f"**PROJECT:**\n{project_context}")

        if supervisor_context and supervisor_context.strip():
            context_sections.append(
                f"**CONTEXT FROM SUPERVISOR:**\n{supervisor_context}"
            )

        full_context = (
            "\n\n".join(context_sections)
            if context_sections
            else "No additional context provided."
        )

        return f"""You are a SUBAGENT executing a focused task. You have access to ALL tools to complete this work.

**YOUR TASK:**
{task_description}

**AVAILABLE CONTEXT:**
{full_context}

**IMPORTANT - YOU ARE ISOLATED:**
- You do NOT have access to the supervisor's conversation history
- You do NOT see previous tool calls or their results
- The context above is ALL the information provided to you
- Use your tools to gather any additional information you need

**EXECUTION APPROACH:**
1. Read the task and context carefully
2. Use tools to gather information (file reads, code searches, etc.)
3. Execute the task completely - be thorough
4. Stream your progress - the user sees your work in real-time
5. Make reasonable assumptions and state them

**TOOL USAGE:**
- Use fetch_file with with_line_numbers=true for precise file editing
- Use code changes tools (add_file_to_changes, update_file_lines, insert_lines, delete_lines) for modifications
- Use show_updated_file and show_diff to display your changes
- Use bash_command for running commands if needed

**OUTPUT:**
- Stream your thinking and work as you go
- End with "## Task Result" containing a concise, actionable summary
- The supervisor will use your Task Result for coordination

Now execute the task completely."""

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
        agent: Agent,
        user_prompt: str,
        agent_type: str = "agent",
        message_history: Optional[List[ModelMessage]] = None,
    ) -> str:
        """Collect streaming response from an agent run"""
        from app.modules.intelligence.tools.reasoning_manager import (
            _get_reasoning_manager,
        )

        full_response = ""
        reasoning_manager = _get_reasoning_manager()

        async with agent.iter(
            user_prompt=user_prompt,
            message_history=message_history or [],
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
                                # Accumulate TextPart content for reasoning dump
                                reasoning_manager.append_content(event.part.content)
                                logger.debug(
                                    f"[{agent_type}] Streaming chunk: {event.part.content[:100]}{'...' if len(event.part.content) > 100 else ''}"
                                )
                            if isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                full_response += event.delta.content_delta
                                # Accumulate TextPartDelta content for reasoning dump
                                reasoning_manager.append_content(
                                    event.delta.content_delta
                                )
                                logger.debug(
                                    f"[{agent_type}] Streaming delta: {event.delta.content_delta[:100]}{'...' if len(event.delta.content_delta) > 100 else ''}"
                                )
                elif Agent.is_end_node(node):
                    # Finalize and save reasoning content
                    reasoning_hash = reasoning_manager.finalize_and_save()
                    if reasoning_hash:
                        logger.info(
                            f"Reasoning content saved with hash: {reasoning_hash}"
                        )
                    break

        return full_response

    async def _stream_subagent_response(
        self,
        agent: Agent,
        user_prompt: str,
        agent_type: str = "subagent",
        message_history: Optional[List[ModelMessage]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream subagent response as it comes in"""
        from app.modules.intelligence.tools.reasoning_manager import (
            _get_reasoning_manager,
        )

        reasoning_manager = _get_reasoning_manager()
        async with agent.iter(
            user_prompt=user_prompt,
            message_history=message_history or [],
            usage_limits=UsageLimits(request_limit=None),
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    # Stream tokens from the model's request
                    try:
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(
                                    event.part, TextPart
                                ):
                                    # Accumulate TextPart content for reasoning dump
                                    reasoning_manager.append_content(event.part.content)
                                    yield ChatAgentResponse(
                                        response=event.part.content,
                                        tool_calls=[],
                                        citations=[],
                                    )
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    # Accumulate TextPartDelta content for reasoning dump
                                    reasoning_manager.append_content(
                                        event.delta.content_delta
                                    )
                                    yield ChatAgentResponse(
                                        response=event.delta.content_delta,
                                        tool_calls=[],
                                        citations=[],
                                    )
                    except Exception as e:
                        error_response = self._handle_stream_error(
                            e, f"{agent_type} model request stream"
                        )
                        if error_response:
                            yield error_response
                        continue

                elif Agent.is_call_tools_node(node):
                    # Stream tool calls and results from subagent
                    async for response in self._process_tool_call_node(node, run.ctx):
                        yield response

                elif Agent.is_end_node(node):
                    logger.info(f"{agent_type} result streamed successfully!!")
                    # Finalize and save reasoning content
                    reasoning_hash = reasoning_manager.finalize_and_save()
                    if reasoning_hash:
                        logger.info(
                            f"Reasoning content saved with hash: {reasoning_hash}"
                        )
                    break

    def _create_delegation_cache_key(self, task_description: str, context: str) -> str:
        """Create a unique cache key for delegation result caching"""
        import hashlib

        content = f"{task_description}::{context}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def _stream_subagent_to_queue(
        self,
        agent_type_str: str,
        task_description: str,
        context: str,
        stream_queue: asyncio.Queue,
        cache_key: str,
    ):
        """Stream subagent response to a queue for real-time streaming.

        This is the ONLY place where the subagent actually executes.
        The result is cached so delegate_function can return it without re-executing.

        NOTE: Due to pydantic_ai's architecture, streaming during tool execution is limited.
        We collect all chunks and store them for yielding when the tool completes.
        The chunks are stored in self._delegation_streamed_content for the tool result handler.
        """
        full_response = ""
        collected_chunks: List[ChatAgentResponse] = []

        try:
            # Convert agent type string to AgentType enum
            agent_type = AgentType(agent_type_str)

            # Create the delegate agent with all tools
            delegate_agent = self._create_agent(agent_type, self._current_context)

            # Build project context (minimal - subagent is isolated)
            project_context = self._create_project_context_info(self._current_context)
            max_project_context_length = 1500
            if len(project_context) > max_project_context_length:
                project_context = project_context[:max_project_context_length] + "..."

            # Create delegation prompt - NO conversation history, just task + context
            full_task = self._create_delegation_prompt(
                task_description,
                project_context,
                context,
            )

            logger.info(
                f"Starting subagent streaming for {agent_type.value} with cache_key={cache_key}"
            )

            # Stream the subagent response and collect all chunks
            async for chunk in self._stream_subagent_response(
                delegate_agent,
                full_task,
                agent_type.value,
                message_history=[],  # No message history - subagent is isolated
            ):
                # Store chunk for later yielding when tool completes
                collected_chunks.append(chunk)
                # Also put in queue for any real-time consumers
                await stream_queue.put(chunk)
                # Collect text for the final result
                if chunk.response:
                    full_response += chunk.response

            # Signal completion
            await stream_queue.put(None)

            # Store all collected chunks for yielding when tool result comes
            self._delegation_streamed_content[cache_key] = collected_chunks
            logger.info(
                f"Collected {len(collected_chunks)} chunks for cache_key={cache_key}"
            )

            # Cache the result for delegate_function to retrieve
            if full_response:
                # Extract task result section
                summary = extract_task_result_from_response(full_response)
                self._delegation_result_cache[cache_key] = (
                    summary if summary else full_response
                )
                logger.info(
                    f"Cached delegation result for {cache_key} (length: {len(self._delegation_result_cache[cache_key])} chars)"
                )
            else:
                self._delegation_result_cache[cache_key] = (
                    "## Task Result\n\nNo output from subagent."
                )

        except Exception as e:
            logger.error(f"Error streaming subagent response: {e}", exc_info=True)
            # Put error response in queue
            error_chunk = self._create_error_response(
                f"*Error in subagent execution: {str(e)}*"
            )
            collected_chunks.append(error_chunk)
            await stream_queue.put(error_chunk)
            await stream_queue.put(None)
            # Store collected chunks (including error)
            self._delegation_streamed_content[cache_key] = collected_chunks
            # Cache error result
            self._delegation_result_cache[cache_key] = (
                f"## Task Result\n\n❌ Error during subagent execution: {str(e)}"
            )

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
                # Finalize and save reasoning content
                from app.modules.intelligence.tools.reasoning_manager import (
                    _get_reasoning_manager,
                )

                reasoning_manager = _get_reasoning_manager()
                reasoning_hash = reasoning_manager.finalize_and_save()
                if reasoning_hash:
                    logger.info(f"Reasoning content saved with hash: {reasoning_hash}")
                break

    async def _consume_queue_chunks(
        self,
        queue: asyncio.Queue,
        timeout: float = 0.1,
        max_chunks: int = 10,
    ) -> tuple[List[ChatAgentResponse], bool]:
        """Consume chunks from a queue with timeout, yielding up to max_chunks

        Returns:
            Tuple of (chunks_list, is_completed) where is_completed is True if
            the stream has finished (None sentinel received)
        """
        chunks: List[ChatAgentResponse] = []
        for _ in range(max_chunks):
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
                if chunk is None:  # Sentinel value indicating completion
                    return chunks, True  # Return chunks and completion flag
                chunks.append(chunk)
            except asyncio.TimeoutError:
                break  # No more chunks available within timeout
        return chunks, False  # Return chunks and not completed

    async def _process_tool_call_node(
        self, node: Any, ctx: Any
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process tool call nodes and yield responses"""
        try:
            async with node.stream(ctx) as handle_stream:
                # Track active delegation streams for this tool call node
                active_streams: Dict[str, asyncio.Queue] = {}
                # Track streaming tasks to ensure they complete
                streaming_tasks: Dict[str, asyncio.Task] = {}

                async for event in handle_stream:
                    # Actively consume chunks from all active delegation streams
                    # This ensures we stream chunks as they arrive, not just at event boundaries
                    for tool_call_id in list(active_streams.keys()):
                        queue = active_streams[tool_call_id]
                        chunks, completed = await self._consume_queue_chunks(
                            queue, timeout=0.05, max_chunks=20
                        )
                        for chunk in chunks:
                            yield chunk
                        if completed:
                            # Stream completed, clean up
                            active_streams.pop(tool_call_id, None)
                            self._active_delegation_streams.pop(tool_call_id, None)
                            # Cancel the streaming task if it's still running
                            if tool_call_id in streaming_tasks:
                                task = streaming_tasks.pop(tool_call_id)
                                if not task.done():
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

                    if isinstance(event, FunctionToolCallEvent):
                        tool_call_id = event.part.tool_call_id or ""
                        tool_name = event.part.tool_name

                        # Yield the tool call event
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[create_tool_call_response(event)],
                            citations=[],
                        )

                        # If this is a delegation tool, start streaming the subagent response
                        if is_delegation_tool(tool_name) and tool_call_id:
                            try:
                                # Extract task info from tool call arguments
                                args_dict = event.part.args_as_dict()
                                task_description = args_dict.get("task_description", "")
                                context = args_dict.get("context", "")
                                agent_type_str = (
                                    extract_agent_type_from_delegation_tool(tool_name)
                                )

                                # Create cache key for coordination with delegate_function
                                cache_key = self._create_delegation_cache_key(
                                    task_description, context
                                )

                                # Store the cache_key -> tool_call_id mapping for later retrieval
                                self._delegation_cache_key_map[tool_call_id] = cache_key

                                # Create a queue for streaming chunks
                                stream_queue: asyncio.Queue = asyncio.Queue()
                                active_streams[tool_call_id] = stream_queue
                                self._active_delegation_streams[tool_call_id] = (
                                    stream_queue
                                )

                                # Start streaming the subagent response in the background
                                # This is the ONLY place the subagent executes - it will cache the result
                                streaming_task = asyncio.create_task(
                                    self._stream_subagent_to_queue(
                                        agent_type_str,
                                        task_description,
                                        context,
                                        stream_queue,
                                        cache_key,
                                    )
                                )
                                streaming_tasks[tool_call_id] = streaming_task

                                # Yield a visual indicator that subagent is starting
                                yield ChatAgentResponse(
                                    response="\n\n---\n🤖 **Subagent Starting...**\n\n",
                                    tool_calls=[],
                                    citations=[],
                                )

                                # Immediately try to consume any early chunks
                                chunks, completed = await self._consume_queue_chunks(
                                    stream_queue, timeout=0.01, max_chunks=5
                                )
                                for chunk in chunks:
                                    yield chunk
                                if completed:
                                    active_streams.pop(tool_call_id, None)
                                    self._active_delegation_streams.pop(
                                        tool_call_id, None
                                    )
                                    streaming_tasks.pop(tool_call_id, None)

                            except Exception as e:
                                logger.warning(
                                    f"Error setting up subagent streaming for {tool_name}: {e}"
                                )
                                # Clean up the queue if there was an error
                                if tool_call_id in active_streams:
                                    del active_streams[tool_call_id]
                                if tool_call_id in self._active_delegation_streams:
                                    del self._active_delegation_streams[tool_call_id]
                                if tool_call_id in streaming_tasks:
                                    task = streaming_tasks.pop(tool_call_id)
                                    if not task.done():
                                        task.cancel()

                    if isinstance(event, FunctionToolResultEvent):
                        tool_call_id = event.result.tool_call_id or ""
                        tool_name = event.result.tool_name or "unknown"

                        # If this was a delegation tool, yield all collected streamed content
                        if is_delegation_tool(tool_name) and tool_call_id:
                            # Get the cache_key for this tool_call_id
                            cache_key = self._delegation_cache_key_map.pop(
                                tool_call_id, None
                            )

                            # Yield all collected chunks from the subagent
                            if (
                                cache_key
                                and cache_key in self._delegation_streamed_content
                            ):
                                collected_chunks = (
                                    self._delegation_streamed_content.pop(cache_key)
                                )
                                logger.info(
                                    f"Yielding {len(collected_chunks)} collected subagent chunks for {tool_call_id}"
                                )
                                for chunk in collected_chunks:
                                    yield chunk
                                # Add a visual separator after subagent output
                                yield ChatAgentResponse(
                                    response="\n\n---\n✅ **Subagent Complete**\n\n",
                                    tool_calls=[],
                                    citations=[],
                                )
                            else:
                                # Fallback: drain any remaining chunks from queue
                                if tool_call_id in active_streams:
                                    queue = active_streams[tool_call_id]
                                    # Drain chunks with multiple attempts to catch all
                                    for _ in range(10):  # Try up to 10 times
                                        chunks, completed = (
                                            await self._consume_queue_chunks(
                                                queue, timeout=0.1, max_chunks=50
                                            )
                                        )
                                        for chunk in chunks:
                                            yield chunk
                                        if completed:
                                            break
                                        # Small delay to allow more chunks to arrive
                                        await asyncio.sleep(0.05)

                            # Clean up streams
                            active_streams.pop(tool_call_id, None)
                            self._active_delegation_streams.pop(tool_call_id, None)

                            # Cancel the streaming task if it's still running
                            if tool_call_id in streaming_tasks:
                                task = streaming_tasks.pop(tool_call_id)
                                if not task.done():
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass
                                streaming_tasks.pop(tool_call_id, None)

                        async for response in self._yield_tool_result_event(event):
                            yield response

                # After all events are processed, drain any remaining chunks
                for tool_call_id in list(active_streams.keys()):
                    queue = active_streams[tool_call_id]
                    # Wait longer for final chunks
                    for _ in range(20):  # Try up to 20 times
                        chunks, completed = await self._consume_queue_chunks(
                            queue, timeout=0.1, max_chunks=50
                        )
                        for chunk in chunks:
                            yield chunk
                        if completed:
                            break
                        await asyncio.sleep(0.05)
                    active_streams.pop(tool_call_id, None)
                    self._active_delegation_streams.pop(tool_call_id, None)

                # Cancel any remaining streaming tasks
                for tool_call_id, task in list(streaming_tasks.items()):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    streaming_tasks.pop(tool_call_id, None)

        except (
            ModelRetry,
            AgentRunError,
            UserError,
        ) as pydantic_error:
            error_str = str(pydantic_error)
            # Check for duplicate tool_result error specifically
            if "tool_result" in error_str.lower() and "multiple" in error_str.lower():
                logger.error(
                    f"Duplicate tool_result error in tool call stream: {pydantic_error}. "
                    f"This indicates pydantic_ai's internal message history has duplicate tool results. "
                    f"This may require restarting the agent run with a fresh context."
                )
                yield self._create_error_response(
                    "*Encountered a message history error. This may require starting a new conversation.*"
                )
            else:
                logger.warning(
                    f"Pydantic-ai error in tool call stream: {pydantic_error}"
                )
                yield self._create_error_response(
                    "*Encountered an issue while calling tools. Trying to recover...*"
                )
        except anyio.WouldBlock:
            logger.warning("Tool call stream would block - continuing...")
        except Exception as e:
            error_str = str(e)
            # Check for duplicate tool_result error
            if "tool_result" in error_str.lower() and "multiple" in error_str.lower():
                logger.error(
                    f"Duplicate tool_result error in tool call stream: {e}. "
                    f"This indicates pydantic_ai's internal message history has duplicate tool results."
                )
                yield self._create_error_response(
                    "*Encountered a message history error. This may require starting a new conversation.*"
                )
            else:
                logger.error(f"Unexpected error in tool call stream: {e}")
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

    def _deduplicate_tools_by_name(self, tools: List[Tool]) -> List[Tool]:
        """Deduplicate tools by name, keeping the first occurrence of each tool name"""
        seen_names = set()
        deduplicated = []
        for tool in tools:
            if tool.name not in seen_names:
                seen_names.add(tool.name)
                deduplicated.append(tool)
            else:
                logger.warning(
                    f"Duplicate tool name '{tool.name}' detected, keeping first occurrence"
                )
        return deduplicated

    def _build_delegate_agent_tools(self) -> List[Tool]:
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
            self._wrap_structured_tools(self.tools)
            + self._wrap_structured_tools(todo_tools)
            + self._wrap_structured_tools(code_changes_tools)
            + self._wrap_structured_tools(requirement_tools)
        )
        return self._deduplicate_tools_by_name(all_tools)

    def _build_supervisor_agent_tools(self) -> List[Tool]:
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
                    function=self._create_delegation_function(agent_type),
                )
            )

        all_tools = (
            self._wrap_structured_tools(self.tools)
            + delegation_tools
            + self._wrap_structured_tools(todo_tools)
            + self._wrap_structured_tools(code_changes_tools)
            + self._wrap_structured_tools(requirement_tools)
        )
        # Deduplicate tools by name before returning
        return self._deduplicate_tools_by_name(all_tools)

    def _create_default_delegate_agents(self) -> Dict[AgentType, AgentConfig]:
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
            history_processors=[self._history_processor],
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
            instrument=True,
            instructions=f"""
# Supervisor Agent

## Core Responsibility

You coordinate work by delegating focused tasks to subagents. Keep your context clean with planning and coordination—subagents handle heavy tool usage.

**Be verbose**: Explain reasoning before tool calls, summarize findings after.

---

## Execution Flow

1. **Plan**: Break down the task, create TODOs
2. **Delegate**: Assign focused tasks to subagents with comprehensive context
3. **Track**: Update TODO status (pending → in_progress → completed)
4. **Adapt**: Update plan based on discoveries
5. **Verify**: Ensure all TODOs complete and objective met

---

## Subagent Delegation (Your Most Powerful Tool)

### Critical: Subagents Are Isolated
- Have ALL your tools except delegation
- Receive ONLY: `task_description` + `context` you provide
- Do NOT get your conversation history or previous tool results
- You receive only their final "## Task Result" summary

### When to Delegate (Use Liberally)
- ✅ **Reasoning**: When you need to pause and think through a problem
- ✅ Any task requiring multiple tool calls
- ✅ Code implementation, debugging, analysis
- ✅ Research and investigation
- ✅ Basically ANY focused task—keep your context clean!

### When NOT to Delegate
- ❌ High-level planning/coordination
- ❌ Final synthesis of multiple subagent results

### Context is ESSENTIAL

**Good context:**
```
"The bug is in app/api/router.py lines 45-67. process_request() calls validate_input() which returns None instead of raising an exception. Error: 'NoneType has no attribute data'. validate_input() is in app/utils/validators.py:23-45. Fix should make it raise ValueError on invalid input."
```

**Bad context:**
```
"Check the router file"  // Too vague!
```

### Parallelization
For independent tasks, call `delegate_to_think_execute` multiple times in the SAME response. All run simultaneously.

---

## Progress Summarization

**When**: After major breakthroughs, every 3-5 significant steps, before complex phases, when context builds up.

**Format**:
```markdown
## 📊 Progress Summary

**Status:** [Where you are]

**Accomplishments:** [Major milestones]

**Findings:** [Critical discoveries with file paths/line numbers]

**Challenges:** [Blockers or issues]

**Next Steps:** [Immediate actions]

**Context to Preserve:** [Key paths, functions, decisions]
```

**Why**: Summaries preserve critical info when detailed history is filtered. Include specific references (file paths, line numbers).

---

## Tool Call Summarization

- **Before**: State what you're doing and why (1-2 sentences)
- **After**: Summarize key findings (2-3 sentences)
- **Why**: Tool results get filtered from history—your summaries preserve context

---

## Code Management

- All changes tracked in code changes manager (persists throughout conversation)
- **Virtual workspace**: Edits are NOT applied to actual repo until published
- **Always fetch with line numbers** before editing: `fetch_file` with `with_line_numbers=true`
- **Preserve indentation**: Match surrounding code exactly
- **Verify edits**: Fetch updated lines to confirm correctness
- **Tools**: `add_file_to_changes`, `update_file_lines`, `replace_in_file`, `insert_lines`, `delete_lines`
- **Review**: `get_session_metadata`, `get_file_from_changes`, `get_file_diff`, `show_diff`
- Prefer targeted updates over full rewrites

---

## Requirement Verification

**At task start**: Use `add_requirements` to document ALL output requirements as markdown list

**Before finalizing**: ALWAYS call `get_requirements` and verify each is met

**Why**: Ensures you deliver exactly what was requested

---

## Reminders

- **Delegate liberally**—every delegation keeps YOUR context cleaner
- **Provide comprehensive context**—subagents are isolated
- **Summarize proactively**—don't wait to be asked
- **Your job**: coordination and synthesis; subagents do heavy lifting
            
            Your Identity:
            Role: {self.config.role}
            Goal: {self.config.goal}
            
            Task instructions:
            {self.config.tasks[0].description}

            {multimodal_instructions}

            CONTEXT: {self._create_supervisor_task_description(ctx)}
            """,
            # result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            # model_settings={"max_tokens": 64000},
            history_processors=[self._history_processor],
        )
        self._supervisor_agent = supervisor_agent
        return supervisor_agent

    def _create_delegation_function(self, agent_type: AgentType):
        """Create a delegation function for a specific agent type.

        IMPORTANT: This function does NOT execute the subagent itself.
        The subagent execution happens in _stream_subagent_to_queue which runs
        in parallel and caches the result. This function waits for that cached result.

        This design ensures:
        1. Subagent only executes once (not twice)
        2. Streaming happens in real-time to the user
        3. Supervisor gets the result after streaming completes
        """

        async def delegate_function(
            ctx: RunContext[None], task_description: str, context: str = ""
        ) -> str:
            """Delegate a task to a subagent for isolated execution.

            CRITICAL: Subagents are ISOLATED - they receive ONLY what you provide here.
            They do NOT get your conversation history or previous tool results.

            Args:
                task_description: Clear, detailed description of the task to execute.
                    Be specific about what you want the subagent to do.
                context: ESSENTIAL context for the subagent. Since subagents are isolated,
                    you MUST provide all relevant information:
                    - File paths and line numbers you've identified
                    - Code snippets relevant to the task
                    - Previous findings or analysis results
                    - Error messages, configuration values, specific details
                    - Everything the subagent needs to work autonomously

                    Example: "Bug in app/api/router.py:45-67. Function process_request()
                    calls validate_input() which returns None. Error: 'NoneType has no
                    attribute data'. validate_input() is in app/utils/validators.py:23-45."

            Returns:
                The task result from the subagent's "## Task Result" section.
                The full subagent work is streamed to the user in real-time.
            """
            try:
                logger.info(
                    f"Delegation function called for {agent_type.value}: {task_description[:100]}..."
                )

                # Create cache key to coordinate with _stream_subagent_to_queue
                cache_key = self._create_delegation_cache_key(task_description, context)

                # Wait for the streaming task to complete and cache the result
                # The actual subagent execution happens in _stream_subagent_to_queue
                # which was started when the tool call event was detected
                max_wait_time = 300  # 5 minutes max wait
                poll_interval = 0.1  # 100ms polling
                waited = 0

                logger.info(f"Waiting for cached result with key={cache_key}")

                while waited < max_wait_time:
                    if cache_key in self._delegation_result_cache:
                        result = self._delegation_result_cache.pop(
                            cache_key
                        )  # Remove from cache
                        logger.info(
                            f"Retrieved cached result for {agent_type.value} "
                            f"(key={cache_key}, length={len(result)} chars)"
                        )
                        return result

                    await asyncio.sleep(poll_interval)
                    waited += poll_interval

                    # Log progress periodically
                    if int(waited) % 10 == 0 and waited > 0:
                        logger.debug(
                            f"Still waiting for delegation result... ({waited}s)"
                        )

                # Timeout - this shouldn't happen normally
                logger.error(
                    f"Timeout waiting for delegation result (key={cache_key}). "
                    f"This may indicate the streaming task failed to start."
                )
                return self._format_delegation_error(
                    agent_type,
                    task_description,
                    "timeout",
                    f"Timed out waiting for subagent result after {max_wait_time}s",
                    "",
                )

            except Exception as e:
                logger.error(
                    f"Error in delegation to {agent_type.value}: {e}", exc_info=True
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
                except Exception as e:
                    logger.error(
                        f"Invalid base64 format for image {attachment_id}: {str(e)}"
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

            except Exception as e:
                logger.error(
                    f"Failed to add image {attachment_id} to content: {str(e)}",
                    exc_info=True,
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
                    except Exception as e:
                        logger.error(
                            f"Invalid base64 format for context image {attachment_id}: {str(e)}"
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
                except Exception as e:
                    logger.error(
                        f"Failed to add context image {attachment_id} to content: {str(e)}",
                        exc_info=True,
                    )
                    continue

        logger.info(
            f"Final multimodal content has {len(content)} elements: 1 text + {len(content)-1} images"
        )
        return content

    def _validate_and_fix_message_history(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """Validate message history and ensure tool calls/results are properly paired.

        Anthropic API requires:
        - Every tool_use block must have a tool_result block in the IMMEDIATELY NEXT message
        - Removes orphaned tool results (results without corresponding calls)
        - Removes orphaned tool calls (calls without corresponding results)
        - Removes tool calls that don't have results in the next message

        This prevents "tool_result without tool_use" and "tool_use without tool_result" errors.
        """
        if not messages:
            return messages

        def _extract_tool_call_ids(msg: ModelMessage) -> set:
            """Extract all tool_call_ids from a message."""
            ids = set()
            parts_to_check = []
            if isinstance(msg, ModelRequest):
                parts_to_check = msg.parts
            elif isinstance(msg, ModelResponse):
                parts_to_check = msg.parts

            for part in parts_to_check:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    if "tool_call_id" in part_dict and part_dict["tool_call_id"]:
                        ids.add(part_dict["tool_call_id"])
                    if "part" in part_dict:
                        part_obj = part_dict["part"]
                        if hasattr(part_obj, "tool_call_id") and part_obj.tool_call_id:
                            ids.add(part_obj.tool_call_id)
                    if "result" in part_dict:
                        result_obj = part_dict["result"]
                        if (
                            hasattr(result_obj, "tool_call_id")
                            and result_obj.tool_call_id
                        ):
                            ids.add(result_obj.tool_call_id)
            return ids

        def _is_tool_call_msg(msg: ModelMessage) -> bool:
            """Check if message contains tool calls (not results)."""
            parts_to_check = []
            if isinstance(msg, ModelRequest):
                parts_to_check = msg.parts
            elif isinstance(msg, ModelResponse):
                parts_to_check = msg.parts

            for part in parts_to_check:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check part_kind first
                    if part_dict.get("part_kind") == "tool-call":
                        return True
                    # Has tool_call_id and tool_name but no result
                    if "tool_call_id" in part_dict and "tool_name" in part_dict:
                        if "result" not in part_dict and "content" not in part_dict:
                            return True
                    if "part" in part_dict:
                        part_obj = part_dict["part"]
                        if hasattr(part_obj, "tool_call_id") and hasattr(
                            part_obj, "tool_name"
                        ):
                            if not hasattr(part_obj, "result"):
                                return True
            return False

        def _is_tool_result_msg(msg: ModelMessage) -> bool:
            """Check if message contains tool results."""
            parts_to_check = []
            if isinstance(msg, ModelRequest):
                parts_to_check = msg.parts
            elif isinstance(msg, ModelResponse):
                parts_to_check = msg.parts

            for part in parts_to_check:
                if hasattr(part, "__dict__"):
                    part_dict = part.__dict__
                    # Check part_kind first
                    if part_dict.get("part_kind") == "tool-return":
                        return True
                    # Has result or content with tool_call_id
                    if "result" in part_dict or (
                        "tool_call_id" in part_dict and "content" in part_dict
                    ):
                        return True
            return False

        # Track tool_call_ids that have been used (tool calls)
        tool_call_ids_with_calls: set = set()
        # Track tool_call_ids that have results
        tool_call_ids_with_results: set = set()
        # Track message metadata
        message_tool_call_ids: list = []

        # First pass: identify all tool calls and results
        for i, msg in enumerate(messages):
            call_ids = _extract_tool_call_ids(msg)
            message_tool_call_ids.append(call_ids)

            if _is_tool_call_msg(msg):
                tool_call_ids_with_calls.update(call_ids)
            elif _is_tool_result_msg(msg):
                tool_call_ids_with_results.update(call_ids)

        # Find orphaned tool calls and results
        orphaned_calls = tool_call_ids_with_calls - tool_call_ids_with_results
        orphaned_results = tool_call_ids_with_results - tool_call_ids_with_calls

        # CRITICAL: Find tool calls that don't have results in the IMMEDIATELY NEXT message
        tool_calls_without_next_result: set = set()
        for i, msg in enumerate(messages):
            if _is_tool_call_msg(msg):
                call_ids = message_tool_call_ids[i]
                if i + 1 < len(messages):
                    next_result_ids = message_tool_call_ids[i + 1]
                    # All call_ids must have results in next message
                    missing = call_ids - next_result_ids
                    if missing:
                        tool_calls_without_next_result.update(missing)
                        logger.debug(
                            f"[Message History Validation] Tool calls at message {i} missing results in next message: {missing}"
                        )
                else:
                    # No next message
                    tool_calls_without_next_result.update(call_ids)

        # Combine all problematic tool_call_ids
        problematic_ids = (
            orphaned_calls | orphaned_results | tool_calls_without_next_result
        )

        if problematic_ids:
            logger.warning(
                f"[Message History Validation] Found {len(problematic_ids)} problematic tool_call_ids: "
                f"orphaned_calls={orphaned_calls}, orphaned_results={orphaned_results}, "
                f"calls_without_next_result={tool_calls_without_next_result}. Removing to prevent API errors."
            )

            # Second pass: identify messages to remove (including their pairs)
            messages_to_skip: set = set()

            for i, msg in enumerate(messages):
                call_ids = message_tool_call_ids[i]
                is_tool_call = _is_tool_call_msg(msg)
                is_tool_result = _is_tool_result_msg(msg)

                if not call_ids:
                    continue

                # CRITICAL: Remove if ANY tool_call_id is problematic
                if any(tid in problematic_ids for tid in call_ids):
                    messages_to_skip.add(i)
                    logger.debug(
                        f"[Message History Validation] Marking message {i} for removal "
                        f"(has problematic tool_call_ids)"
                    )

                    # Also remove paired messages
                    if is_tool_call and i + 1 < len(messages):
                        messages_to_skip.add(i + 1)
                        logger.debug(
                            f"[Message History Validation] Also removing paired result at message {i+1}"
                        )
                    if is_tool_result and i > 0 and _is_tool_call_msg(messages[i - 1]):
                        messages_to_skip.add(i - 1)
                        logger.debug(
                            f"[Message History Validation] Also removing paired call at message {i-1}"
                        )

            # Build filtered messages
            filtered_messages = [
                msg for i, msg in enumerate(messages) if i not in messages_to_skip
            ]

            # Safety check
            if not filtered_messages and messages:
                logger.error(
                    "[Message History Validation] All messages removed! Keeping original."
                )
                return messages

            return filtered_messages

        return messages

    async def _prepare_multimodal_message_history(
        self, ctx: ChatContext
    ) -> List[ModelMessage]:
        """Prepare message history with multimodal support.

        CRITICAL: This method now prioritizes using compressed message history from previous runs.
        It retrieves compressed messages from the history processor's internal storage.
        Otherwise, it falls back to rebuilding from ctx.history (text strings).

        This ensures that compression benefits are preserved across multiple agent runs within
        the same execution context.
        """
        # Try to get compressed history from the history processor
        # The processor stores compressed messages keyed by run_id from RunContext
        # We need to check if we have a stored run_id from a previous run
        # For now, we'll use a simple approach: check if processor has any stored history
        # and use the most recent one (since we're in the same execution context)

        # Access the processor instance from the history processor function
        processor = getattr(self._history_processor, "processor", None)
        if processor:
            # Get all stored compressed histories (there should be at most one per run)
            stored_keys = list(processor._last_compressed_output.keys())
            if stored_keys:
                # Use the most recent key (last one added)
                latest_key = stored_keys[-1]
                compressed_history = processor.get_compressed_history(latest_key)
                if compressed_history:
                    logger.info(
                        f"Using compressed message history from processor: {len(compressed_history)} messages "
                        f"(key: {latest_key})"
                    )
                    # Validate the compressed history before using it
                    return self._validate_and_fix_message_history(compressed_history)

        # Fallback: rebuild from ctx.history if no compressed history available
        logger.debug("No compressed history found, rebuilding from ctx.history")
        history_messages = []

        # Limit history to prevent token bloat (max 8 messages or ~50k tokens estimated)
        # This prevents "prompt too long" errors and reduces chance of duplicate tool_result issues
        max_history_messages = 8
        limited_history = (
            ctx.history[-max_history_messages:]
            if len(ctx.history) > max_history_messages
            else ctx.history
        )

        if len(ctx.history) > max_history_messages:
            logger.warning(
                f"Message history truncated from {len(ctx.history)} to {len(limited_history)} messages "
                f"to prevent token limit issues"
            )

        for msg in limited_history:
            # For now, keep history as text-only to avoid token bloat
            # Images are only added to the current query
            history_messages.append(ModelResponse([TextPart(content=str(msg))]))

        # Validate and fix message history to ensure tool calls/results are paired
        history_messages = self._validate_and_fix_message_history(history_messages)

        return history_messages

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow with multi-agent coordination"""
        logger.info(
            f"Running pydantic multi-agent system {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Reset todo manager, code changes manager, and requirement manager for this agent run to ensure isolation
        from app.modules.intelligence.tools.todo_management_tool import (
            _reset_todo_manager,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            _reset_code_changes_manager,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            _reset_requirement_manager,
        )

        _reset_todo_manager()
        _reset_code_changes_manager()
        _reset_requirement_manager()
        logger.info(
            "🔄 Reset todo manager, code changes manager, and requirement manager for new agent run"
        )

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

            # Validate message history before sending to model (single validation)
            message_history = self._validate_and_fix_message_history(message_history)

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

            # Note: Compressed messages are now stored internally by the history processor
            # The processor automatically stores them when it processes messages
            # No need to manually capture here - they'll be retrieved in _prepare_multimodal_message_history

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(
                f"Error in standard multi-agent run method: {str(e)}", exc_info=True
            )
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

            # Note: Compressed messages are stored internally by the history processor

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(
                f"Error in multimodal multi-agent run method: {str(e)}", exc_info=True
            )
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

        # Reset todo manager, code changes manager, and requirement manager for this agent run to ensure isolation
        from app.modules.intelligence.tools.todo_management_tool import (
            _reset_todo_manager,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            _reset_code_changes_manager,
        )
        from app.modules.intelligence.tools.requirement_verification_tool import (
            _reset_requirement_manager,
        )

        _reset_todo_manager()
        _reset_code_changes_manager()
        _reset_requirement_manager()
        logger.info(
            "🔄 Reset todo manager, code changes manager, and requirement manager for new agent run"
        )

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

            # Validate message history before sending to model
            message_history = self._validate_and_fix_message_history(message_history)

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
                # Store the supervisor run so delegation functions can access its message history
                self._current_supervisor_run = run
                try:
                    async for response in self._process_agent_run_nodes(
                        run, "multimodal multi-agent"
                    ):
                        yield response
                finally:
                    # Note: For streaming runs, compressed messages are handled by history processor
                    # Clear the reference when done
                    self._current_supervisor_run = None

        except Exception as e:
            logger.error(
                f"Error in multimodal multi-agent stream: {str(e)}", exc_info=True
            )
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
                # Use _prepare_multimodal_message_history to get compressed history if available
                message_history = await self._prepare_multimodal_message_history(ctx)
                message_history = self._validate_and_fix_message_history(
                    message_history
                )

                async with supervisor_agent.run_mcp_servers():
                    async with supervisor_agent.iter(
                        user_prompt=ctx.query,
                        message_history=message_history,
                        usage_limits=UsageLimits(
                            request_limit=None
                        ),  # No request limit for long-running tasks
                    ) as run:
                        # Store the supervisor run so delegation functions can access its message history
                        self._current_supervisor_run = run
                        try:
                            async for response in self._process_agent_run_nodes(
                                run, "multi-agent"
                            ):
                                yield response
                        finally:
                            # Note: For streaming runs, compressed messages are handled by history processor
                            # Clear the reference when done
                            self._current_supervisor_run = None

            except (
                TimeoutError,
                anyio.WouldBlock,
                ModelHTTPError,
                Exception,
            ) as mcp_error:
                error_detail = f"{type(mcp_error).__name__}: {str(mcp_error)}"
                error_str = str(mcp_error).lower()

                # Check for specific error types
                if isinstance(mcp_error, ModelHTTPError):
                    error_body = getattr(mcp_error, "body", {})
                    error_message = (
                        error_body.get("error", {}).get("message", "")
                        if isinstance(error_body, dict)
                        else str(error_body)
                    )

                    # Check for duplicate tool_result error
                    if (
                        "tool_result" in error_message.lower()
                        and "multiple" in error_message.lower()
                    ):
                        logger.error(
                            f"Duplicate tool_result error detected in ModelHTTPError: {error_message}. "
                            f"This indicates pydantic_ai's message history has duplicate tool results. "
                            f"This may be caused by retries or error recovery. The message history may need to be cleared."
                        )
                    # Check for token limit error
                    elif (
                        "too long" in error_message.lower()
                        or "maximum" in error_message.lower()
                    ):
                        logger.error(
                            f"Token limit exceeded: {error_message}. "
                            f"Message history is too large. Consider reducing history size or starting a new conversation."
                        )

                logger.warning(
                    f"MCP server initialization failed in stream: {error_detail}",
                    exc_info=True,
                )
                # Check if it's a JSON parsing error
                if "json" in error_str or "parse" in error_str:
                    logger.error(
                        f"JSON parsing error during MCP server initialization in stream - MCP server may be returning malformed or incomplete JSON. Full traceback:\n{traceback.format_exc()}"
                    )
                logger.info("Continuing without MCP servers...")

                # Fallback without MCP servers - use compressed history if available
                message_history = await self._prepare_multimodal_message_history(ctx)
                message_history = self._validate_and_fix_message_history(
                    message_history
                )

                async with supervisor_agent.iter(
                    user_prompt=ctx.query,
                    message_history=message_history,
                    usage_limits=UsageLimits(
                        request_limit=None
                    ),  # No request limit for long-running tasks
                ) as run:
                    # Store the supervisor run so delegation functions can access its message history
                    self._current_supervisor_run = run
                    try:
                        async for response in self._process_agent_run_nodes(
                            run, "multi-agent"
                        ):
                            yield response
                    finally:
                        # Clear the reference when done
                        self._current_supervisor_run = None

        except Exception as e:
            logger.error(
                f"Error in standard multi-agent stream: {str(e)}", exc_info=True
            )
            yield ChatAgentResponse(
                response="\n\n*An error occurred during multi-agent streaming*\n\n",
                tool_calls=[],
                citations=[],
            )
