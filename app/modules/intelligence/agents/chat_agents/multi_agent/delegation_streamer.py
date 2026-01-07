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
# These are generous timeouts to allow complex tasks to complete
# Event-driven keepalive mechanisms prevent getting stuck even with long timeouts
AGENT_ITER_TIMEOUT = 600.0  # 10 minutes max for entire agent run
NODE_TIMEOUT = 300.0  # 5 minutes max per node (allows for complex tool executions)
EVENT_TIMEOUT = 120.0  # 2 minutes max between LLM stream events
TOOL_EXECUTION_TIMEOUT = 300.0  # 5 minutes max for tool execution

# Keepalive/heartbeat intervals for event-driven monitoring
KEEPALIVE_INTERVAL = 30.0  # Emit keepalive every 30 seconds during long operations
PROGRESS_LOG_INTERVAL = 15.0  # Log progress every 15 seconds


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
        """Run agent with overall timeout wrapper and event-driven monitoring."""

        async def _run_agent():
            async with agent.iter(
                user_prompt=user_prompt,
                message_history=message_history or [],
                usage_limits=UsageLimits(request_limit=None),
            ) as run:
                logger.info(
                    f"[SUBAGENT] agent.iter() started (agent_type={agent_type})"
                )

                # Use manual iteration with timeout instead of async for
                # to prevent hanging when pydantic-ai is slow to yield nodes
                node_iter = run.__aiter__()
                # Generous timeout for node iteration - allows complex operations
                node_iteration_timeout = 180.0  # 3 minutes max to get next node
                consecutive_node_timeouts = 0
                max_consecutive_node_timeouts = 2  # Allow 2 consecutive timeouts
                last_progress_log = asyncio.get_event_loop().time()
                agent_start_time = last_progress_log

                while True:
                    current_time = asyncio.get_event_loop().time()

                    # Progress logging
                    if current_time - last_progress_log >= PROGRESS_LOG_INTERVAL:
                        total_elapsed = current_time - agent_start_time
                        logger.info(
                            f"[SUBAGENT] 📊 Progress: agent_type={agent_type}, "
                            f"nodes={execution_state['node_count']}, elapsed={total_elapsed:.1f}s"
                        )
                        last_progress_log = current_time

                    # Get next node with timeout to prevent hanging
                    try:
                        node = await asyncio.wait_for(
                            node_iter.__anext__(), timeout=node_iteration_timeout
                        )
                        consecutive_node_timeouts = 0  # Reset on success

                    except StopAsyncIteration:
                        logger.info(
                            f"[SUBAGENT] Node iteration completed normally (agent_type={agent_type})"
                        )
                        break

                    except asyncio.TimeoutError:
                        consecutive_node_timeouts += 1
                        total_elapsed = (
                            asyncio.get_event_loop().time() - agent_start_time
                        )

                        if consecutive_node_timeouts >= max_consecutive_node_timeouts:
                            logger.error(
                                f"[SUBAGENT] Node iteration TIMEOUT #{consecutive_node_timeouts} after {node_iteration_timeout}s "
                                f"(agent_type={agent_type}, total_elapsed={total_elapsed:.1f}s). "
                                f"Max consecutive timeouts reached. Breaking gracefully."
                            )
                            yield ChatAgentResponse(
                                response=f"\n*Agent timed out waiting for next step after {total_elapsed:.0f}s - using partial results*\n",
                                tool_calls=[],
                                citations=[],
                            )
                            break
                        else:
                            logger.warning(
                                f"[SUBAGENT] Node iteration timeout #{consecutive_node_timeouts} "
                                f"(agent_type={agent_type}, elapsed={total_elapsed:.1f}s). Retrying..."
                            )
                            # Emit keepalive and continue
                            yield ChatAgentResponse(
                                response="",  # Silent keepalive
                                tool_calls=[],
                                citations=[],
                            )
                            continue

                    except asyncio.CancelledError:
                        logger.info(
                            f"[SUBAGENT] Node iteration cancelled (agent_type={agent_type})"
                        )
                        raise

                    except Exception as e:
                        logger.error(
                            f"[SUBAGENT] Node iteration error (agent_type={agent_type}): {e}",
                            exc_info=True,
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Error getting next step: {str(e)[:100]}*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        break

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
                        logger.warning(
                            f"[SUBAGENT] Node #{node_count} timeout after {node_elapsed:.1f}s - continuing to next node"
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Step #{node_count} timed out after {node_elapsed:.0f}s - continuing*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        # Continue to next node instead of failing completely
                        continue

                    except asyncio.CancelledError:
                        logger.info(f"[SUBAGENT] Node #{node_count} cancelled")
                        raise

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
            # CRITICAL: Generator cleanup can hang if pydantic-ai's context manager
            # is stuck waiting for something. Use a timeout to prevent infinite hangs.
            try:
                await asyncio.wait_for(gen.aclose(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[SUBAGENT] Generator cleanup timed out after 10s (agent_type={agent_type}). "
                    f"This may indicate pydantic-ai's agent.iter() context manager is stuck."
                )
            except asyncio.CancelledError:
                logger.debug(
                    f"[SUBAGENT] Generator cleanup cancelled (agent_type={agent_type})"
                )
            except Exception as cleanup_error:
                logger.warning(
                    f"[SUBAGENT] Generator cleanup error (agent_type={agent_type}): {cleanup_error}"
                )

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
                last_keepalive_time = last_event_time
                stream_start_time = last_event_time
                consecutive_timeouts = 0
                max_consecutive_timeouts = (
                    3  # Allow 3 consecutive timeouts before giving up
                )
                stream_iter = request_stream.__aiter__()

                while True:
                    current_time = asyncio.get_event_loop().time()

                    # Emit keepalive if it's been a while since last yield
                    # This prevents upstream timeouts during long-running operations
                    if current_time - last_keepalive_time >= KEEPALIVE_INTERVAL:
                        elapsed_total = current_time - stream_start_time
                        time_since_event = current_time - last_event_time
                        logger.info(
                            f"[SUBAGENT] Node #{node_count}: 💓 keepalive "
                            f"(elapsed={elapsed_total:.1f}s, since_event={time_since_event:.1f}s, "
                            f"events={event_count})"
                        )
                        yield ChatAgentResponse(
                            response="",  # Empty keepalive
                            tool_calls=[],
                            citations=[],
                        )
                        last_keepalive_time = current_time

                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=EVENT_TIMEOUT
                        )
                        last_event_time = asyncio.get_event_loop().time()
                        last_keepalive_time = last_event_time  # Reset keepalive timer
                        consecutive_timeouts = 0  # Reset timeout counter on success

                    except StopAsyncIteration:
                        logger.info(
                            f"[SUBAGENT] Node #{node_count}: stream completed "
                            f"(events={event_count}, yields={yield_count})"
                        )
                        break

                    except asyncio.TimeoutError:
                        consecutive_timeouts += 1
                        elapsed_total = (
                            asyncio.get_event_loop().time() - stream_start_time
                        )
                        time_since_event = (
                            asyncio.get_event_loop().time() - last_event_time
                        )

                        if consecutive_timeouts >= max_consecutive_timeouts:
                            logger.error(
                                f"[SUBAGENT] Node #{node_count}: {consecutive_timeouts} consecutive event timeouts "
                                f"(events={event_count}, yields={yield_count}, elapsed={elapsed_total:.1f}s). "
                                f"LLM stream appears stuck. Breaking out gracefully."
                            )
                            yield ChatAgentResponse(
                                response=f"\n*Stream timeout after {elapsed_total:.0f}s - using partial results*\n",
                                tool_calls=[],
                                citations=[],
                            )
                            break
                        else:
                            logger.warning(
                                f"[SUBAGENT] Node #{node_count}: event timeout #{consecutive_timeouts} "
                                f"(events={event_count}, yields={yield_count}, since_event={time_since_event:.1f}s). "
                                f"Retrying... (max={max_consecutive_timeouts})"
                            )
                            # Yield a keepalive to prevent upstream timeouts
                            yield ChatAgentResponse(
                                response="",  # Silent keepalive
                                tool_calls=[],
                                citations=[],
                            )
                            last_keepalive_time = asyncio.get_event_loop().time()
                            continue  # Retry getting the next event

                    except asyncio.CancelledError:
                        logger.info(f"[SUBAGENT] Node #{node_count}: stream cancelled")
                        raise

                    except Exception as e:
                        # Handle unexpected errors gracefully
                        logger.error(
                            f"[SUBAGENT] Node #{node_count}: unexpected error getting next event: {e}",
                            exc_info=True,
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Stream error: {str(e)[:100]}*\n",
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

                        # Yield keepalive every 15 non-text events to prevent upstream timeouts
                        if event_count - yield_count >= 15:
                            yield ChatAgentResponse(
                                response="",
                                tool_calls=[],
                                citations=[],
                            )
                            yield_count = event_count
                            last_keepalive_time = asyncio.get_event_loop().time()

            finally:
                # Always properly exit the stream context with timeout
                # to prevent hanging if pydantic-ai's stream context is stuck
                try:
                    await asyncio.wait_for(
                        stream_ctx.__aexit__(None, None, None), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[SUBAGENT] Node #{node_count}: stream context exit timed out after 5s"
                    )
                except asyncio.CancelledError:
                    logger.debug(
                        f"[SUBAGENT] Node #{node_count}: stream context exit cancelled"
                    )
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
        """Execute tool call with timeout protection and keepalive mechanism."""

        tool_ctx = None
        current_tool_name = "unknown"

        try:
            # Timeout for entering the tool stream context (generous for slow tools)
            try:
                tool_ctx = node.stream(run.ctx)
                tool_stream = await asyncio.wait_for(
                    tool_ctx.__aenter__(),
                    timeout=60.0,  # 60s to enter context
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[SUBAGENT] Node #{node_count}: tool stream context timeout (60s)"
                )
                yield ChatAgentResponse(
                    response="\n*Tool initialization timeout - the tool took too long to start*\n",
                    tool_calls=[],
                    citations=[],
                )
                return
            except Exception as e:
                logger.error(
                    f"[SUBAGENT] Node #{node_count}: tool stream context error: {e}"
                )
                yield ChatAgentResponse(
                    response=f"\n*Tool initialization error: {str(e)[:100]}*\n",
                    tool_calls=[],
                    citations=[],
                )
                return

            tool_start = asyncio.get_event_loop().time()  # Initialize before try block
            try:
                logger.info(f"[SUBAGENT] Node #{node_count}: tool stream entered")
                last_keepalive_time = tool_start
                event_count = 0

                async for event in tool_stream:
                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - tool_start
                    event_count += 1

                    # Emit keepalive during long-running tool executions
                    if current_time - last_keepalive_time >= KEEPALIVE_INTERVAL:
                        logger.info(
                            f"[SUBAGENT] Node #{node_count}: 💓 tool keepalive "
                            f"(tool={current_tool_name}, elapsed={elapsed:.1f}s, events={event_count})"
                        )
                        yield ChatAgentResponse(
                            response="",  # Silent keepalive
                            tool_calls=[],
                            citations=[],
                        )
                        last_keepalive_time = current_time

                    # Check for overall tool timeout
                    if elapsed > TOOL_EXECUTION_TIMEOUT:
                        logger.warning(
                            f"[SUBAGENT] Node #{node_count}: tool execution timeout after {elapsed:.1f}s "
                            f"(tool={current_tool_name})"
                        )
                        yield ChatAgentResponse(
                            response=f"\n*Tool '{current_tool_name}' timed out after {elapsed:.0f}s*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        break

                    if isinstance(event, FunctionToolCallEvent):
                        current_tool_name = event.part.tool_name
                        logger.info(f"[SUBAGENT] Executing tool: {current_tool_name}")
                        yield ChatAgentResponse(
                            response=f"\n*Executing {current_tool_name}...*\n",
                            tool_calls=[],
                            citations=[],
                        )
                        last_keepalive_time = current_time  # Reset keepalive timer

                    elif isinstance(event, FunctionToolResultEvent):
                        tool_elapsed = current_time - tool_start
                        result_tool_name = event.result.tool_name
                        logger.info(
                            f"[SUBAGENT] Tool completed: {result_tool_name} "
                            f"in {tool_elapsed:.1f}s"
                        )
                        # Log slow tool warnings
                        if tool_elapsed > 60.0:
                            logger.warning(
                                f"[SUBAGENT] Slow tool execution: {result_tool_name} "
                                f"took {tool_elapsed:.1f}s"
                            )

            except asyncio.CancelledError:
                logger.info(f"[SUBAGENT] Node #{node_count}: tool execution cancelled")
                raise

            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - tool_start
                logger.error(
                    f"[SUBAGENT] Node #{node_count}: error during tool execution "
                    f"(tool={current_tool_name}, elapsed={elapsed:.1f}s): {e}",
                    exc_info=True,
                )
                yield ChatAgentResponse(
                    response=f"\n*Tool '{current_tool_name}' error: {str(e)[:100]}*\n",
                    tool_calls=[],
                    citations=[],
                )

            finally:
                # Tool context exit with timeout to prevent hanging
                if tool_ctx is not None:
                    try:
                        await asyncio.wait_for(
                            tool_ctx.__aexit__(None, None, None), timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[SUBAGENT] Node #{node_count}: tool context exit timed out after 10s"
                        )
                    except asyncio.CancelledError:
                        logger.debug(
                            f"[SUBAGENT] Node #{node_count}: tool context exit cancelled"
                        )
                    except Exception as e:
                        logger.debug(
                            f"[SUBAGENT] Tool stream exit error (ignored): {e}"
                        )

        except asyncio.CancelledError:
            logger.info(f"[SUBAGENT] Node #{node_count}: tool call cancelled")
            raise

        except Exception as e:
            logger.error(
                f"[SUBAGENT] Node #{node_count}: unexpected tool error: {e}",
                exc_info=True,
            )
            yield ChatAgentResponse(
                response=f"\n*Unexpected tool error: {str(e)[:100]}*\n",
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
