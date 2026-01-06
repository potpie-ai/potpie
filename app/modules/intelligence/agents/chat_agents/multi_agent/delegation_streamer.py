"""Delegation streamer for streaming subagent responses.

Robust architecture designed to:
1. Never get stuck - all operations have timeouts
2. Return early on errors
3. Provide meaningful feedback on failures
4. Clean up properly on exit
5. Bubble errors up to supervisor with structured information
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional, Any
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    TextPart,
    ModelMessage,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.usage import UsageLimits

from app.modules.intelligence.agents.chat_agent import ChatAgentResponse
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Timeout constants for subagent operations
AGENT_ITER_TIMEOUT = 180.0  # 3 minutes max for entire agent run
NODE_TIMEOUT = 120.0  # 2 minutes max per node
EVENT_TIMEOUT = 30.0  # 30 seconds max between events
TOOL_EXECUTION_TIMEOUT = 120.0  # 2 minutes max for tool execution


class SubagentErrorType(Enum):
    """Types of errors that can occur in subagent execution."""

    TIMEOUT = "timeout"
    STREAM_TIMEOUT = "stream_timeout"
    TOOL_TIMEOUT = "tool_timeout"
    API_ERROR = "api_error"
    TOOL_ERROR = "tool_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class SubagentError:
    """Structured error information from subagent execution."""

    error_type: SubagentErrorType
    message: str
    agent_type: str
    elapsed_seconds: float
    node_count: int = 0
    partial_response: str = ""
    recoverable: bool = False
    suggestion: str = ""

    def to_supervisor_message(self) -> str:
        """Format error for supervisor to understand and potentially retry."""
        error_header = f"## ❌ Subagent Error ({self.error_type.value})\n\n"

        details = [
            f"**Agent:** {self.agent_type}",
            f"**Error:** {self.message}",
            f"**Duration:** {self.elapsed_seconds:.1f}s",
        ]

        if self.node_count > 0:
            details.append(f"**Nodes processed:** {self.node_count}")

        if self.suggestion:
            details.append(f"\n**Suggestion:** {self.suggestion}")

        if self.partial_response:
            # Include partial response if available (truncated)
            partial = self.partial_response[:500]
            if len(self.partial_response) > 500:
                partial += "... [truncated]"
            details.append(f"\n**Partial output:**\n```\n{partial}\n```")

        if self.recoverable:
            details.append(
                "\n*This error may be recoverable. You can retry with a simpler task.*"
            )

        return error_header + "\n".join(details)


class SubagentTimeoutError(Exception):
    """Raised when a subagent operation times out."""

    def __init__(self, message: str, error_info: Optional[SubagentError] = None):
        super().__init__(message)
        self.error_info = error_info


class SubagentExecutionError(Exception):
    """Raised when a subagent encounters an unrecoverable error."""

    def __init__(self, message: str, error_info: Optional[SubagentError] = None):
        super().__init__(message)
        self.error_info = error_info


# Special marker for error responses that the consumer can detect
ERROR_MARKER = "[[SUBAGENT_ERROR]]"


def is_subagent_error(response: str) -> bool:
    """Check if a subagent response indicates an error.

    Args:
        response: The subagent response text

    Returns:
        True if the response contains error indicators
    """
    if not response:
        return False

    error_indicators = [
        ERROR_MARKER,
        "## ❌ Subagent Error",
        "*Subagent error:",
        "*Stream timeout",
        "*Tool execution timeout*",
        "*Model request timeout*",
        "agent timed out after",
        "agent was cancelled",
    ]

    return any(indicator in response for indicator in error_indicators)


def extract_error_type_from_response(response: str) -> Optional[SubagentErrorType]:
    """Extract the error type from a subagent error response.

    Args:
        response: The subagent response text

    Returns:
        The SubagentErrorType if detected, None otherwise
    """
    if not response:
        return None

    # Check for specific error type indicators
    error_type_patterns = {
        SubagentErrorType.TIMEOUT: ["timed out", "timeout", "(timeout)"],
        SubagentErrorType.STREAM_TIMEOUT: ["stream_timeout", "Stream timeout"],
        SubagentErrorType.TOOL_TIMEOUT: ["tool_timeout", "Tool execution timeout"],
        SubagentErrorType.API_ERROR: ["api_error", "API issue"],
        SubagentErrorType.TOOL_ERROR: ["tool_error", "Tool failed"],
        SubagentErrorType.CANCELLED: ["cancelled", "was cancelled"],
    }

    for error_type, patterns in error_type_patterns.items():
        if any(pattern.lower() in response.lower() for pattern in patterns):
            return error_type

    if is_subagent_error(response):
        return SubagentErrorType.UNKNOWN

    return None


class DelegationStreamer:
    """Handles streaming of subagent responses with robust error handling."""

    def __init__(
        self,
        handle_stream_error: Any,
        create_error_response: Any,
        process_tool_call_node: Any = None,
    ):
        self.handle_stream_error = handle_stream_error
        self.create_error_response = create_error_response
        self.process_tool_call_node = process_tool_call_node

    async def stream_subagent_response(
        self,
        agent: Agent,
        user_prompt: str,
        agent_type: str = "subagent",
        message_history: Optional[List[ModelMessage]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream subagent response with comprehensive error handling and timeouts.

        This generator is designed to NEVER get stuck:
        - Overall execution timeout prevents infinite runs
        - Per-node timeouts catch stuck operations
        - Per-event timeouts catch slow/hanging streams
        - All errors result in early return with structured error response
        - Errors are formatted for supervisor to understand and act upon
        """
        from app.modules.intelligence.tools.reasoning_manager import (
            _get_reasoning_manager,
        )

        reasoning_manager = _get_reasoning_manager()
        start_time = asyncio.get_event_loop().time()

        # Track execution state for error reporting
        execution_state = {
            "node_count": 0,
            "partial_response": "",
            "last_error": None,
        }

        logger.info(
            f"[SUBAGENT] Starting for agent_type={agent_type}, "
            f"prompt_length={len(user_prompt)}, timeout={AGENT_ITER_TIMEOUT}s"
        )

        try:
            # Wrap entire agent execution in a timeout
            async for response in self._run_agent_with_timeout(
                agent,
                user_prompt,
                agent_type,
                message_history,
                reasoning_manager,
                execution_state,
            ):
                # Track partial response for error reporting
                if response.response and not response.response.startswith(ERROR_MARKER):
                    execution_state["partial_response"] += response.response
                yield response

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"[SUBAGENT] ⚠️ TIMEOUT after {elapsed:.1f}s (agent_type={agent_type}, "
                f"nodes={execution_state['node_count']})"
            )
            yield self._create_timeout_response(
                agent_type,
                elapsed,
                partial_response=execution_state["partial_response"],
                node_count=execution_state["node_count"],
            )

        except asyncio.CancelledError:
            logger.warning(f"[SUBAGENT] Cancelled (agent_type={agent_type})")
            yield self._create_cancelled_response(agent_type)
            raise

        except SubagentTimeoutError as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"[SUBAGENT] Subagent timeout after {elapsed:.1f}s (agent_type={agent_type}): {e}"
            )
            if e.error_info:
                yield ChatAgentResponse(
                    response=f"{ERROR_MARKER}\n{e.error_info.to_supervisor_message()}",
                    tool_calls=[],
                    citations=[],
                )
            else:
                yield self._create_timeout_response(
                    agent_type,
                    elapsed,
                    partial_response=execution_state["partial_response"],
                    node_count=execution_state["node_count"],
                )

        except SubagentExecutionError as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"[SUBAGENT] Execution error after {elapsed:.1f}s (agent_type={agent_type}): {e}"
            )
            if e.error_info:
                yield ChatAgentResponse(
                    response=f"{ERROR_MARKER}\n{e.error_info.to_supervisor_message()}",
                    tool_calls=[],
                    citations=[],
                )
            else:
                yield self._create_error_response(
                    agent_type,
                    str(e),
                    error_type=SubagentErrorType.UNKNOWN,
                    elapsed=elapsed,
                    partial_response=execution_state["partial_response"],
                    node_count=execution_state["node_count"],
                )

        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            error_str = str(e)

            # Classify error type based on exception
            if "timeout" in error_str.lower():
                error_type = SubagentErrorType.TIMEOUT
            elif "api" in error_str.lower() or "rate" in error_str.lower():
                error_type = SubagentErrorType.API_ERROR
            else:
                error_type = SubagentErrorType.UNKNOWN

            logger.error(
                f"[SUBAGENT] Error after {elapsed:.1f}s (agent_type={agent_type}, "
                f"type={error_type.value}): {e}",
                exc_info=True,
            )
            yield self._create_error_response(
                agent_type,
                error_str,
                error_type=error_type,
                elapsed=elapsed,
                partial_response=execution_state["partial_response"],
                node_count=execution_state["node_count"],
            )

        finally:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"[SUBAGENT] Finished after {elapsed:.1f}s (agent_type={agent_type}, "
                f"nodes={execution_state['node_count']}, "
                f"response_length={len(execution_state['partial_response'])})"
            )

    async def _run_agent_with_timeout(
        self,
        agent: Agent,
        user_prompt: str,
        agent_type: str,
        message_history: Optional[List[ModelMessage]],
        reasoning_manager: Any,
        execution_state: dict,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run agent with overall timeout wrapper."""

        async def _run_agent():
            async with agent.iter(
                user_prompt=user_prompt,
                message_history=message_history or [],
                usage_limits=UsageLimits(request_limit=None),
            ) as run:
                logger.info(
                    f"[SUBAGENT] agent.iter() started (agent_type={agent_type})"
                )

                async for node in run:
                    execution_state["node_count"] += 1
                    node_count = execution_state["node_count"]
                    node_start = asyncio.get_event_loop().time()

                    try:
                        async for response in self._process_node_with_timeout(
                            node, run, node_count, agent_type, reasoning_manager
                        ):
                            yield response

                        node_elapsed = asyncio.get_event_loop().time() - node_start
                        if node_elapsed > 10:  # Log if node took > 10s
                            logger.info(
                                f"[SUBAGENT] Node #{node_count} completed in {node_elapsed:.1f}s"
                            )

                    except asyncio.TimeoutError:
                        node_elapsed = asyncio.get_event_loop().time() - node_start
                        logger.error(
                            f"[SUBAGENT] Node #{node_count} TIMEOUT after {node_elapsed:.1f}s"
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Node timeout after {node_elapsed:.0f}s - continuing...*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        # Continue to next node instead of failing completely
                        continue

                    except Exception as e:
                        logger.error(
                            f"[SUBAGENT] Node #{node_count} error: {e}",
                            exc_info=True,
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Error in node: {str(e)[:100]}*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        continue

                    # Check if this is an end node
                    if Agent.is_end_node(node):
                        reasoning_hash = reasoning_manager.finalize_and_save()
                        if reasoning_hash:
                            logger.info(f"[SUBAGENT] Reasoning saved: {reasoning_hash}")
                        break

        # Use asyncio.wait_for with overall timeout
        gen = _run_agent()
        deadline = asyncio.get_event_loop().time() + AGENT_ITER_TIMEOUT

        try:
            async for response in gen:
                # Check if we've exceeded the overall deadline
                if asyncio.get_event_loop().time() > deadline:
                    raise asyncio.TimeoutError("Agent execution exceeded deadline")
                yield response
        finally:
            await gen.aclose()

    async def _process_node_with_timeout(
        self,
        node: Any,
        run: Any,
        node_count: int,
        agent_type: str,
        reasoning_manager: Any,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process a single node with timeout protection."""

        if Agent.is_model_request_node(node):
            logger.info(f"[SUBAGENT] Node #{node_count}: model_request")

            async for response in self._stream_model_request(
                node, run, node_count, agent_type, reasoning_manager
            ):
                yield response

        elif Agent.is_call_tools_node(node):
            logger.info(f"[SUBAGENT] Node #{node_count}: call_tools")

            async for response in self._execute_tool_call(
                node, run, node_count, agent_type
            ):
                yield response

        elif Agent.is_end_node(node):
            logger.info(f"[SUBAGENT] Node #{node_count}: end")
            # End node is handled by caller

        else:
            logger.debug(f"[SUBAGENT] Node #{node_count}: other (skipped)")

    async def _stream_model_request(
        self,
        node: Any,
        run: Any,
        node_count: int,
        agent_type: str,
        reasoning_manager: Any,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream model request with per-event timeout."""

        try:
            # Timeout for entering the stream context
            try:
                stream_ctx = node.stream(run.ctx)
                request_stream = await asyncio.wait_for(
                    stream_ctx.__aenter__(), timeout=NODE_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"[SUBAGENT] Node #{node_count}: timeout entering stream")
                yield ChatAgentResponse(
                    response="\n*Model request timeout*\n",
                    tool_calls=[],
                    citations=[],
                )
                return

            try:
                logger.info(f"[SUBAGENT] Node #{node_count}: stream entered")
                event_count = 0
                yield_count = 0
                last_event_time = asyncio.get_event_loop().time()
                stream_iter = request_stream.__aiter__()

                while True:
                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=EVENT_TIMEOUT
                        )
                        last_event_time = asyncio.get_event_loop().time()

                    except StopAsyncIteration:
                        logger.info(
                            f"[SUBAGENT] Node #{node_count}: stream completed "
                            f"(events={event_count}, yields={yield_count})"
                        )
                        break

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[SUBAGENT] Node #{node_count}: event timeout "
                            f"(events={event_count}, yields={yield_count})"
                        )
                        yield ChatAgentResponse(
                            response="\n*Stream timeout - continuing...*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        break

                    event_count += 1

                    # Handle text events
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, TextPart
                    ):
                        reasoning_manager.append_content(event.part.content)
                        yield_count += 1
                        yield ChatAgentResponse(
                            response=event.part.content,
                            tool_calls=[],
                            citations=[],
                        )

                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        reasoning_manager.append_content(event.delta.content_delta)
                        yield_count += 1
                        yield ChatAgentResponse(
                            response=event.delta.content_delta,
                            tool_calls=[],
                            citations=[],
                        )

                    # Handle non-text events (tool calls) - yield keepalive
                    elif isinstance(event, (PartStartEvent, PartDeltaEvent)):
                        # Log tool call events periodically
                        if event_count % 20 == 1:
                            part_type = type(
                                getattr(event, "part", None)
                                or getattr(event, "delta", None)
                            ).__name__
                            logger.debug(
                                f"[SUBAGENT] Node #{node_count}: tool event "
                                f"#{event_count} ({part_type})"
                            )

                        # Yield keepalive every 15 non-text events
                        if event_count - yield_count >= 15:
                            yield ChatAgentResponse(
                                response="",
                                tool_calls=[],
                                citations=[],
                            )
                            yield_count = event_count

            finally:
                # Always properly exit the stream context
                try:
                    await stream_ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"[SUBAGENT] Stream exit error (ignored): {e}")

        except Exception as e:
            logger.error(
                f"[SUBAGENT] Node #{node_count}: stream error: {e}",
                exc_info=True,
            )
            yield ChatAgentResponse(
                response=f"\n*Stream error: {str(e)[:100]}*\n",
                tool_calls=[],
                citations=[],
            )

    async def _execute_tool_call(
        self,
        node: Any,
        run: Any,
        node_count: int,
        agent_type: str,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Execute tool call with timeout protection."""

        try:
            # Timeout for entering the tool stream context
            try:
                tool_ctx = node.stream(run.ctx)
                tool_stream = await asyncio.wait_for(
                    tool_ctx.__aenter__(), timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"[SUBAGENT] Node #{node_count}: tool stream timeout")
                yield ChatAgentResponse(
                    response="\n*Tool execution timeout*\n",
                    tool_calls=[],
                    citations=[],
                )
                return

            try:
                logger.info(f"[SUBAGENT] Node #{node_count}: tool stream entered")
                tool_start = asyncio.get_event_loop().time()

                async for event in tool_stream:
                    # Check for overall tool timeout
                    if (
                        asyncio.get_event_loop().time() - tool_start
                        > TOOL_EXECUTION_TIMEOUT
                    ):
                        logger.warning(
                            f"[SUBAGENT] Node #{node_count}: tool execution timeout"
                        )
                        yield ChatAgentResponse(
                            response="\n*Tool execution took too long*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        break

                    if isinstance(event, FunctionToolCallEvent):
                        tool_name = event.part.tool_name
                        logger.info(f"[SUBAGENT] Executing tool: {tool_name}")
                        yield ChatAgentResponse(
                            response=f"\n*Executing {tool_name}...*\n",
                            tool_calls=[],
                            citations=[],
                        )

                    elif isinstance(event, FunctionToolResultEvent):
                        elapsed = asyncio.get_event_loop().time() - tool_start
                        logger.info(
                            f"[SUBAGENT] Tool completed: {event.result.tool_name} "
                            f"in {elapsed:.1f}s"
                        )

            finally:
                try:
                    await tool_ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"[SUBAGENT] Tool stream exit error (ignored): {e}")

        except Exception as e:
            logger.error(
                f"[SUBAGENT] Node #{node_count}: tool error: {e}",
                exc_info=True,
            )
            yield ChatAgentResponse(
                response=f"\n*Tool error: {str(e)[:100]}*\n",
                tool_calls=[],
                citations=[],
            )

    def _create_timeout_response(
        self,
        agent_type: str,
        elapsed: float,
        partial_response: str = "",
        node_count: int = 0,
    ) -> ChatAgentResponse:
        """Create a structured timeout error response that supervisor can understand."""
        error_info = SubagentError(
            error_type=SubagentErrorType.TIMEOUT,
            message=f"Subagent execution timed out after {elapsed:.0f}s",
            agent_type=agent_type,
            elapsed_seconds=elapsed,
            node_count=node_count,
            partial_response=partial_response,
            recoverable=True,
            suggestion="Try breaking down the task into smaller steps, or simplify the request.",
        )

        return ChatAgentResponse(
            response=f"{ERROR_MARKER}\n{error_info.to_supervisor_message()}",
            tool_calls=[],
            citations=[],
        )

    def _create_cancelled_response(self, agent_type: str) -> ChatAgentResponse:
        """Create a structured cancellation response."""
        error_info = SubagentError(
            error_type=SubagentErrorType.CANCELLED,
            message="Subagent execution was cancelled",
            agent_type=agent_type,
            elapsed_seconds=0,
            recoverable=False,
        )

        return ChatAgentResponse(
            response=f"{ERROR_MARKER}\n{error_info.to_supervisor_message()}",
            tool_calls=[],
            citations=[],
        )

    def _create_error_response(
        self,
        agent_type: str,
        error: str,
        error_type: SubagentErrorType = SubagentErrorType.UNKNOWN,
        elapsed: float = 0,
        partial_response: str = "",
        node_count: int = 0,
    ) -> ChatAgentResponse:
        """Create a structured error response that supervisor can understand and act upon."""
        # Truncate error message to avoid huge responses
        error_msg = error[:500] + "..." if len(error) > 500 else error

        # Determine if error is recoverable based on type
        recoverable = error_type in [
            SubagentErrorType.STREAM_TIMEOUT,
            SubagentErrorType.TOOL_TIMEOUT,
            SubagentErrorType.API_ERROR,
        ]

        # Provide suggestions based on error type
        suggestions = {
            SubagentErrorType.TIMEOUT: "Try a simpler or more focused task.",
            SubagentErrorType.STREAM_TIMEOUT: "The model response was incomplete. Try again or simplify.",
            SubagentErrorType.TOOL_TIMEOUT: "A tool took too long. The integration may be slow.",
            SubagentErrorType.API_ERROR: "There was an API issue. Retry may help.",
            SubagentErrorType.TOOL_ERROR: "A tool failed. Check if the tool parameters are correct.",
            SubagentErrorType.UNKNOWN: "An unexpected error occurred.",
        }

        error_info = SubagentError(
            error_type=error_type,
            message=error_msg,
            agent_type=agent_type,
            elapsed_seconds=elapsed,
            node_count=node_count,
            partial_response=partial_response,
            recoverable=recoverable,
            suggestion=suggestions.get(error_type, ""),
        )

        return ChatAgentResponse(
            response=f"{ERROR_MARKER}\n{error_info.to_supervisor_message()}",
            tool_calls=[],
            citations=[],
        )

    @staticmethod
    async def collect_agent_streaming_response(
        agent: Agent,
        user_prompt: str,
        agent_type: str = "agent",
        message_history: Optional[List[ModelMessage]] = None,
    ) -> str:
        """Collect streaming response from an agent run (non-streaming collection)."""
        from app.modules.intelligence.tools.reasoning_manager import (
            _get_reasoning_manager,
        )

        full_response = ""
        reasoning_manager = _get_reasoning_manager()

        async def _collect():
            nonlocal full_response
            async with agent.iter(
                user_prompt=user_prompt,
                message_history=message_history or [],
                usage_limits=UsageLimits(request_limit=None),
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent) and isinstance(
                                    event.part, TextPart
                                ):
                                    full_response += event.part.content
                                    reasoning_manager.append_content(event.part.content)
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    full_response += event.delta.content_delta
                                    reasoning_manager.append_content(
                                        event.delta.content_delta
                                    )
                    elif Agent.is_end_node(node):
                        reasoning_hash = reasoning_manager.finalize_and_save()
                        if reasoning_hash:
                            logger.info(f"Reasoning saved: {reasoning_hash}")
                        break

        try:
            await asyncio.wait_for(_collect(), timeout=AGENT_ITER_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("[SUBAGENT] collect_agent_streaming_response timeout")
            full_response += "\n*Response timed out*\n"

        return full_response
