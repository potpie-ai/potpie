"""Delegation manager for handling agent delegation state and functions"""

import asyncio
from typing import Dict, List, Callable, Optional, Any
from pydantic_ai import RunContext

from .delegation_streamer import (
    ERROR_MARKER,
    is_subagent_error,
)
from .utils.delegation_utils import (
    AgentType,
    create_delegation_cache_key,
    format_delegation_error,
    extract_task_result_from_response,
)
from .utils.context_utils import create_project_context_info
from .utils.tool_call_stream_manager import ToolCallStreamManager
from app.modules.intelligence.agents.chat_agent import ChatContext, ChatAgentResponse
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class DelegationManager:
    """Manages delegation state and provides delegation functions"""

    def __init__(
        self,
        create_delegate_agent: Callable[[AgentType, ChatContext], Any],
        delegation_streamer: Any,
        create_error_response: Callable[[str], ChatAgentResponse],
    ):
        """Initialize the delegation manager

        Args:
            create_delegate_agent: Function to create a delegate agent instance
            delegation_streamer: DelegationStreamer instance for streaming responses
            create_error_response: Function to create error responses
        """
        self.create_delegate_agent = create_delegate_agent
        self.delegation_streamer = delegation_streamer
        self.create_error_response = create_error_response
        self.tool_call_stream_manager = ToolCallStreamManager()

        # Track active streaming tasks for delegation tools (tool_call_id -> queue)
        self._active_delegation_streams: Dict[str, asyncio.Queue] = {}
        # Cache results from streaming to avoid duplicate execution (task_key -> result)
        self._delegation_result_cache: Dict[str, str] = {}
        # Store streamed content for each delegation (cache_key -> list of chunks)
        self._delegation_streamed_content: Dict[str, List[ChatAgentResponse]] = {}
        # Map tool_call_id to cache_key for retrieving streamed content
        self._delegation_cache_key_map: Dict[str, str] = {}
        # Track active streaming tasks by cache_key to detect if they're running
        self._active_streaming_tasks: Dict[str, asyncio.Task] = {}

    def create_delegation_function(self, agent_type: AgentType) -> Callable:
        """Create a delegation function for a specific agent type.

        IMPORTANT: This function does NOT execute the subagent itself.
        The subagent execution happens in stream_subagent_to_queue which runs
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
                # Create cache key to coordinate with stream_subagent_to_queue
                cache_key = create_delegation_cache_key(task_description, context)

                # Wait for the streaming task to complete and cache the result
                # The actual subagent execution happens in stream_subagent_to_queue
                # which was started when the tool call event was detected
                max_wait_time = 300  # 5 minutes max wait
                poll_interval = 0.1  # 100ms polling
                waited = 0
                last_task_status_log = 0

                logger.info(
                    f"[DELEGATE_FUNCTION] Waiting for result (agent_type={agent_type.value}, "
                    f"cache_key={cache_key}, task_description={task_description[:100]}...)"
                )

                # Check if streaming task exists and is running
                streaming_task = self._active_streaming_tasks.get(cache_key)
                if streaming_task:
                    logger.info(
                        f"[DELEGATE_FUNCTION] Found active streaming task for cache_key={cache_key}, "
                        f"task_done={streaming_task.done()}"
                    )
                else:
                    logger.warning(
                        f"[DELEGATE_FUNCTION] No active streaming task found for cache_key={cache_key}. "
                        f"This may indicate the task hasn't started yet or cache key mismatch."
                    )

                while waited < max_wait_time:
                    # Check if result is cached
                    if cache_key in self._delegation_result_cache:
                        result = self._delegation_result_cache.pop(
                            cache_key
                        )  # Remove from cache
                        logger.info(
                            f"[DELEGATE_FUNCTION] Result found and returning (agent_type={agent_type.value}, "
                            f"cache_key={cache_key}, result_length={len(result)} chars)"
                        )
                        # Clean up task tracking
                        self._active_streaming_tasks.pop(cache_key, None)
                        return result

                    # Check if streaming task exists and has failed
                    streaming_task = self._active_streaming_tasks.get(cache_key)
                    if streaming_task and streaming_task.done():
                        # Task completed but no result cached - this is an error
                        try:
                            await streaming_task  # This will raise if task failed
                        except Exception as task_error:
                            logger.error(
                                f"[DELEGATE_FUNCTION] Streaming task failed for cache_key={cache_key}: {task_error}",
                                exc_info=True,
                            )
                            # Cache error result to prevent infinite wait
                            error_result = format_delegation_error(
                                agent_type,
                                task_description,
                                type(task_error).__name__,
                                str(task_error),
                                "",
                            )
                            self._delegation_result_cache[cache_key] = error_result
                            self._active_streaming_tasks.pop(cache_key, None)
                            return error_result

                        # Task completed successfully but no result - this shouldn't happen
                        # Wait a bit more in case result is being cached
                        if waited > 1.0:  # After 1 second, something is wrong
                            logger.error(
                                f"[DELEGATE_FUNCTION] Streaming task completed but no result cached "
                                f"for cache_key={cache_key}. This indicates a bug in stream_subagent_to_queue."
                            )
                            # Cache error result to prevent infinite wait
                            error_result = format_delegation_error(
                                agent_type,
                                task_description,
                                "no_result",
                                "Streaming task completed but no result was cached",
                                "",
                            )
                            self._delegation_result_cache[cache_key] = error_result
                            self._active_streaming_tasks.pop(cache_key, None)
                            return error_result

                    await asyncio.sleep(poll_interval)
                    waited += poll_interval

                    # Log every 5 seconds with task status
                    if int(waited) - last_task_status_log >= 5:
                        last_task_status_log = int(waited)
                        task_status = "unknown"
                        if cache_key in self._active_streaming_tasks:
                            task = self._active_streaming_tasks[cache_key]
                            task_status = (
                                f"done={task.done()}, cancelled={task.cancelled()}"
                            )
                        else:
                            task_status = "not_found"
                        logger.warning(
                            f"[DELEGATE_FUNCTION] Still waiting... (agent_type={agent_type.value}, "
                            f"cache_key={cache_key}, waited={int(waited)}s, task_status={task_status})"
                        )

                # Timeout - this shouldn't happen normally
                logger.error(
                    f"[DELEGATE_FUNCTION] Timeout waiting for delegation result (key={cache_key}). "
                    f"This may indicate the streaming task failed to start or never completed."
                )
                # Clean up task tracking
                self._active_streaming_tasks.pop(cache_key, None)
                return format_delegation_error(
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
                return format_delegation_error(
                    agent_type, task_description, type(e).__name__, str(e), ""
                )

        return delegate_function

    async def stream_subagent_to_queue(
        self,
        agent_type_str: str,
        task_description: str,
        context: str,
        stream_queue: asyncio.Queue,
        cache_key: str,
        current_context: ChatContext,
        call_id: Optional[str] = None,
    ):
        """Stream subagent response to a queue for real-time streaming and Redis streams.

        This is the ONLY place where the subagent actually executes.
        The result is cached so delegate_function can return it without re-executing.

        Args:
            agent_type_str: Type of agent to delegate to
            task_description: Task description for the subagent
            context: Context string for the subagent
            stream_queue: AsyncIO queue for backward compatibility
            cache_key: Cache key for this delegation
            current_context: Current chat context
            call_id: Optional tool call ID for Redis streaming (if provided, streams to Redis)
        """
        full_response = ""
        collected_chunks: List[ChatAgentResponse] = []
        subagent_error_occurred = False  # Track if subagent reported an error

        try:
            # Convert agent type string to AgentType enum
            agent_type = AgentType(agent_type_str)

            # Create the delegate agent with all tools
            delegate_agent = self.create_delegate_agent(agent_type, current_context)

            # Build project context (minimal - subagent is isolated)
            project_context = create_project_context_info(current_context)
            max_project_context_length = 1500
            if len(project_context) > max_project_context_length:
                project_context = project_context[:max_project_context_length] + "..."

            # Create delegation prompt - NO conversation history, just task + context
            from .utils.delegation_utils import create_delegation_prompt

            full_task = create_delegation_prompt(
                task_description,
                project_context,
                context,
            )

            # Log task details for debugging
            logger.info(
                f"[SUBAGENT STREAM] Starting stream for agent_type={agent_type.value}, "
                f"cache_key={cache_key}, call_id={call_id}"
            )
            logger.info(
                f"[SUBAGENT STREAM] Task description: {task_description[:200]}{'...' if len(task_description) > 200 else ''}"
            )
            logger.info(
                f"[SUBAGENT STREAM] Context provided: {context[:200]}{'...' if len(context) > 200 else ''}"
            )

            # Stream the subagent response and collect all chunks
            logger.info(
                f"[SUBAGENT STREAM] Starting to iterate over subagent response (agent_type={agent_type.value}, "
                f"cache_key={cache_key})"
            )

            chunk_count = 0
            stream_start_time = asyncio.get_event_loop().time()
            last_chunk_time = stream_start_time
            # Align with AGENT_ITER_TIMEOUT from delegation_streamer (180s)
            # Add buffer for cleanup and error handling
            stream_timeout = 200.0  # 3 min 20s total timeout
            chunk_timeout = (
                45.0  # 45 second timeout between chunks (EVENT_TIMEOUT + buffer)
            )

            # Get the async generator
            stream_gen = self.delegation_streamer.stream_subagent_response(
                delegate_agent,
                full_task,
                agent_type.value,
                message_history=[],  # No message history - subagent is isolated
            )

            # Stream with timeout protection
            try:
                logger.info(
                    f"[SUBAGENT STREAM] Starting chunk iteration loop (agent_type={agent_type.value}, "
                    f"cache_key={cache_key}, call_id={call_id}, "
                    f"chunk_timeout={chunk_timeout}s, stream_timeout={stream_timeout}s)"
                )
                loop_iteration = 0
                last_heartbeat_time = stream_start_time
                heartbeat_interval = 10.0  # Log heartbeat every 10 seconds

                while True:
                    loop_iteration += 1
                    current_loop_time = asyncio.get_event_loop().time()

                    # Heartbeat log every 10 seconds to show we're still in the loop
                    if current_loop_time - last_heartbeat_time >= heartbeat_interval:
                        elapsed = current_loop_time - stream_start_time
                        time_since_last_chunk = current_loop_time - last_chunk_time
                        logger.info(
                            f"[SUBAGENT STREAM] 💓 HEARTBEAT: Loop iteration #{loop_iteration} "
                            f"(agent_type={agent_type.value}, chunks={chunk_count}, "
                            f"elapsed={elapsed:.1f}s, time_since_last_chunk={time_since_last_chunk:.1f}s, "
                            f"cache_key={cache_key})"
                        )
                        last_heartbeat_time = current_loop_time

                    if loop_iteration % 10 == 0:  # Log every 10 iterations
                        elapsed = asyncio.get_event_loop().time() - stream_start_time
                        logger.debug(
                            f"[SUBAGENT STREAM] Chunk loop iteration #{loop_iteration} "
                            f"(agent_type={agent_type.value}, chunks={chunk_count}, elapsed={elapsed:.1f}s)"
                        )

                    # Check for overall timeout
                    elapsed = asyncio.get_event_loop().time() - stream_start_time
                    if elapsed > stream_timeout:
                        logger.error(
                            f"[SUBAGENT STREAM] Stream timeout after {elapsed:.1f}s (agent_type={agent_type.value}, "
                            f"cache_key={cache_key}, call_id={call_id}). Received {chunk_count} chunks, "
                            f"{len(full_response)} chars. Caching partial result."
                        )
                        break

                    # Get next chunk with timeout
                    # Note: Keep-alive messages during tool execution prevent this from timing out
                    time_before_chunk = asyncio.get_event_loop().time()
                    time_since_last_chunk = time_before_chunk - last_chunk_time
                    if (
                        time_since_last_chunk > 5.0
                    ):  # Log if waiting more than 5 seconds
                        logger.info(
                            f"[SUBAGENT STREAM] ⚠️ Waiting for next chunk (agent_type={agent_type.value}, "
                            f"chunk_count={chunk_count}, time_since_last={time_since_last_chunk:.2f}s, "
                            f"timeout={chunk_timeout}s, cache_key={cache_key})"
                        )
                    else:
                        logger.debug(
                            f"[SUBAGENT STREAM] Waiting for next chunk (agent_type={agent_type.value}, "
                            f"chunk_count={chunk_count}, time_since_last={time_since_last_chunk:.2f}s, "
                            f"timeout={chunk_timeout}s)"
                        )
                    try:
                        chunk = await asyncio.wait_for(
                            stream_gen.__anext__(), timeout=chunk_timeout
                        )
                        chunk_wait_time = (
                            asyncio.get_event_loop().time() - time_before_chunk
                        )
                        last_chunk_time = asyncio.get_event_loop().time()
                        if chunk_wait_time > 5.0:  # Log if wait was long
                            logger.info(
                                f"[SUBAGENT STREAM] Received chunk after {chunk_wait_time:.2f}s wait "
                                f"(agent_type={agent_type.value}, chunk_count={chunk_count + 1}, cache_key={cache_key})"
                            )
                        else:
                            logger.debug(
                                f"[SUBAGENT STREAM] Received chunk after {chunk_wait_time:.2f}s wait "
                                f"(agent_type={agent_type.value}, chunk_count={chunk_count + 1})"
                            )
                    except StopAsyncIteration:
                        # Stream completed normally
                        elapsed = asyncio.get_event_loop().time() - stream_start_time
                        logger.info(
                            f"[SUBAGENT STREAM] Stream completed normally after {elapsed:.1f}s "
                            f"(agent_type={agent_type.value}, cache_key={cache_key}, "
                            f"chunk_count={chunk_count})"
                        )
                        break
                    except asyncio.TimeoutError:
                        time_since_last_chunk = (
                            asyncio.get_event_loop().time() - last_chunk_time
                        )
                        elapsed_total = (
                            asyncio.get_event_loop().time() - stream_start_time
                        )
                        logger.warning(
                            f"[SUBAGENT STREAM] ⚠️ Chunk timeout after {chunk_timeout}s "
                            f"(agent_type={agent_type.value}, cache_key={cache_key}, call_id={call_id}). "
                            f"Last chunk was #{chunk_count} ({time_since_last_chunk:.1f}s ago). "
                            f"Total elapsed: {elapsed_total:.1f}s. "
                            f"Caching partial result with {len(full_response)} chars. "
                            f"This may indicate the subagent generator is stuck waiting for a tool or node."
                        )
                        break

                    chunk_count += 1
                    logger.debug(
                        f"[SUBAGENT STREAM] Processing chunk #{chunk_count} (agent_type={agent_type.value}, "
                        f"cache_key={cache_key})"
                    )

                    # Filter out tool calls - we only want text responses from subagents
                    # Create a clean chunk with only text content (no tool calls)
                    text_only_chunk = ChatAgentResponse(
                        response=chunk.response or "",
                        tool_calls=[],  # Don't stream subagent tool calls to frontend
                        citations=chunk.citations or [],
                    )

                    # Store chunk for later yielding when tool completes
                    collected_chunks.append(text_only_chunk)
                    # Also put in queue for any real-time consumers (backward compatibility)
                    # Use put_nowait to avoid blocking - if queue is full, log warning but continue
                    try:
                        stream_queue.put_nowait(text_only_chunk)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"[SUBAGENT STREAM] Queue full, dropping chunk (agent_type={agent_type.value}, "
                            f"cache_key={cache_key}, call_id={call_id}). This should not happen with unbounded queues."
                        )
                    except Exception as queue_error:
                        logger.warning(
                            f"[SUBAGENT STREAM] Error putting chunk in queue: {queue_error}"
                        )

                    # Check if this chunk contains an error from the subagent
                    is_error_chunk = chunk.response and is_subagent_error(
                        chunk.response
                    )
                    if is_error_chunk:
                        logger.warning(
                            f"[SUBAGENT STREAM] ⚠️ Error chunk received from subagent "
                            f"(agent_type={agent_type.value}, cache_key={cache_key})"
                        )
                        # Mark that we received an error
                        subagent_error_occurred = True

                    # Log chunk response for debugging
                    # NOTE: We only stream text responses, not tool calls from subagents
                    if chunk.response:
                        # Log full text content for debugging (truncate if very long)
                        text_preview = (
                            chunk.response[:1000]
                            if len(chunk.response) > 1000
                            else chunk.response
                        )
                        log_level = "warning" if is_error_chunk else "info"
                        getattr(logger, log_level)(
                            f"[SUBAGENT STREAM] Chunk #{chunk_count} received (agent_type={agent_type.value}, "
                            f"call_id={call_id}, length={len(chunk.response)} chars, error={is_error_chunk}):\n"
                            f"--- TEXT START ---\n{text_preview}{'... [truncated]' if len(chunk.response) > 1000 else ''}\n"
                            f"--- TEXT END ---"
                        )

                    # Publish to Redis stream if call_id is provided
                    # Only publish text content, not tool calls
                    if call_id and chunk.response:
                        try:
                            self.tool_call_stream_manager.publish_stream_part(
                                call_id=call_id,
                                stream_part=chunk.response,
                                is_complete=False,
                            )
                        except Exception as redis_error:
                            logger.warning(
                                f"Failed to publish to Redis stream for call_id {call_id}: {redis_error}"
                            )

                    # Collect text for the final result
                    if chunk.response:
                        full_response += chunk.response

            except Exception as stream_error:
                logger.error(
                    f"[SUBAGENT STREAM] Error in stream iteration (agent_type={agent_type.value}, "
                    f"cache_key={cache_key}): {stream_error}",
                    exc_info=True,
                )
                # Will be handled by outer exception handler
                raise

            # If we broke out due to timeout, cache partial result
            if cache_key not in self._delegation_result_cache:
                elapsed = asyncio.get_event_loop().time() - stream_start_time
                if elapsed >= stream_timeout or (
                    chunk_count > 0 and elapsed >= chunk_timeout
                ):
                    logger.warning(
                        f"[SUBAGENT STREAM] Caching partial result due to timeout (agent_type={agent_type.value}, "
                        f"cache_key={cache_key}, elapsed={elapsed:.1f}s, chunks={chunk_count})"
                    )
                    partial_result = (
                        full_response
                        if full_response
                        else "## Task Result\n\n⚠️ Subagent execution timed out or was interrupted."
                    )
                    summary = extract_task_result_from_response(partial_result)
                    self._delegation_result_cache[cache_key] = (
                        summary if summary else partial_result
                    )
                    # Put timeout message in queue
                    timeout_chunk = self.create_error_response(
                        f"*Subagent execution timed out after {elapsed:.1f}s. Using partial results.*"
                    )
                    collected_chunks.append(timeout_chunk)
                    await stream_queue.put(timeout_chunk)
                    await stream_queue.put(None)

            # Log full response for debugging
            logger.info(
                f"[SUBAGENT STREAM] Stream completed (agent_type={agent_type.value}, "
                f"cache_key={cache_key}, call_id={call_id}, "
                f"response_length={len(full_response)} chars, chunks={chunk_count})"
            )
            if full_response:
                # Log full response with clear markers
                logger.info(
                    f"[SUBAGENT STREAM] ========== FULL SUBAGENT RESPONSE START ==========\n"
                    f"agent_type={agent_type.value}, call_id={call_id}, cache_key={cache_key}\n"
                    f"--- RESPONSE CONTENT ---\n{full_response}\n"
                    f"--- END RESPONSE CONTENT ---\n"
                    f"========== FULL SUBAGENT RESPONSE END =========="
                )
            else:
                logger.warning(
                    f"[SUBAGENT STREAM] ⚠️ Full response is empty (agent_type={agent_type.value}, "
                    f"call_id={call_id}, cache_key={cache_key}, chunks={chunk_count})"
                )

            # Publish final complete response to Redis stream if call_id is provided
            if call_id:
                try:
                    logger.info(
                        f"[SUBAGENT STREAM] Publishing final complete response to Redis: "
                        f"call_id={call_id}, response_length={len(full_response)}"
                    )
                    self.tool_call_stream_manager.publish_complete(
                        call_id=call_id,
                        tool_response=full_response,
                    )
                    logger.info(
                        f"[SUBAGENT STREAM] Successfully published final complete response to Redis: call_id={call_id}"
                    )
                except Exception as redis_error:
                    logger.error(
                        f"[SUBAGENT STREAM] Failed to publish final response to Redis stream for call_id {call_id}: {redis_error}",
                        exc_info=True,
                    )

            # Store all collected chunks for yielding when tool result comes
            self._delegation_streamed_content[cache_key] = collected_chunks

            # Cache the result for delegate_function to retrieve
            if full_response:
                # Check if the response contains an error from the subagent
                if subagent_error_occurred or is_subagent_error(full_response):
                    # Extract the error message for supervisor
                    error_content = full_response
                    if ERROR_MARKER in full_response:
                        # Remove the marker and keep the formatted error
                        error_content = full_response.replace(ERROR_MARKER, "").strip()

                    logger.warning(
                        f"[SUBAGENT STREAM] Caching error result for supervisor "
                        f"(agent_type={agent_type.value}, cache_key={cache_key})"
                    )
                    # Cache the full error response so supervisor can understand what happened
                    self._delegation_result_cache[cache_key] = error_content
                else:
                    # Extract task result section for successful responses
                    summary = extract_task_result_from_response(full_response)
                    logger.info(
                        f"[SUBAGENT STREAM] Extracted task result (agent_type={agent_type.value}, "
                        f"cache_key={cache_key}):\n{summary}"
                    )
                    self._delegation_result_cache[cache_key] = (
                        summary if summary else full_response
                    )
            else:
                logger.warning(
                    f"[SUBAGENT STREAM] No response from subagent (agent_type={agent_type.value}, "
                    f"cache_key={cache_key})"
                )
                self._delegation_result_cache[cache_key] = (
                    "## Task Result\n\n⚠️ No output from subagent. The task may have failed silently."
                )

            logger.info(
                f"[SUBAGENT STREAM] Successfully cached result for cache_key={cache_key} "
                f"(error={subagent_error_occurred})"
            )

        except Exception as e:
            logger.error(
                f"[SUBAGENT STREAM] Error streaming subagent response (agent_type={agent_type_str}, "
                f"cache_key={cache_key}, call_id={call_id}): {e}",
                exc_info=True,
            )
            # Put error response in queue
            error_message = f"*Error in subagent execution: {str(e)}*"
            error_chunk = self.create_error_response(error_message)
            collected_chunks.append(error_chunk)
            await stream_queue.put(error_chunk)
            await stream_queue.put(None)
            logger.error(f"[SUBAGENT STREAM] Error response sent: {error_message}")

            # Publish error to Redis stream if call_id is provided
            if call_id:
                try:
                    error_message = f"*Error in subagent execution: {str(e)}*"
                    self.tool_call_stream_manager.publish_stream_part(
                        call_id=call_id,
                        stream_part=error_message,
                        is_complete=True,
                        tool_response=error_message,
                    )
                    self.tool_call_stream_manager.publish_complete(
                        call_id=call_id,
                        tool_response=error_message,
                    )
                except Exception as redis_error:
                    logger.warning(
                        f"Failed to publish error to Redis stream for call_id {call_id}: {redis_error}"
                    )

            # Store collected chunks (including error)
            self._delegation_streamed_content[cache_key] = collected_chunks
            # CRITICAL: Always cache error result to prevent delegate_function from waiting forever
            error_result = format_delegation_error(
                AgentType(agent_type_str),
                task_description,
                type(e).__name__,
                str(e),
                "",
            )
            self._delegation_result_cache[cache_key] = error_result
            logger.error(
                f"[SUBAGENT STREAM] Cached error result for cache_key={cache_key} to prevent deadlock"
            )
        finally:
            # Always clean up task tracking
            self._active_streaming_tasks.pop(cache_key, None)
            logger.info(
                f"[SUBAGENT STREAM] Cleaned up task tracking for cache_key={cache_key}"
            )

    def get_delegation_result(self, cache_key: str) -> Optional[str]:
        """Get a cached delegation result"""
        return self._delegation_result_cache.get(cache_key)

    def get_streamed_content(self, cache_key: str) -> Optional[List[ChatAgentResponse]]:
        """Get streamed content for a delegation"""
        return self._delegation_streamed_content.get(cache_key)

    def pop_streamed_content(self, cache_key: str) -> Optional[List[ChatAgentResponse]]:
        """Get and remove streamed content for a delegation"""
        return self._delegation_streamed_content.pop(cache_key, None)

    def register_active_stream(self, tool_call_id: str, queue: asyncio.Queue):
        """Register an active delegation stream"""
        self._active_delegation_streams[tool_call_id] = queue

    def get_active_stream(self, tool_call_id: str) -> Optional[asyncio.Queue]:
        """Get an active delegation stream"""
        return self._active_delegation_streams.get(tool_call_id)

    def remove_active_stream(self, tool_call_id: str):
        """Remove an active delegation stream"""
        self._active_delegation_streams.pop(tool_call_id, None)

    def map_cache_key(self, tool_call_id: str, cache_key: str):
        """Map a tool_call_id to a cache_key"""
        self._delegation_cache_key_map[tool_call_id] = cache_key

    def get_cache_key(self, tool_call_id: str) -> Optional[str]:
        """Get the cache_key for a tool_call_id"""
        return self._delegation_cache_key_map.get(tool_call_id)

    def pop_cache_key(self, tool_call_id: str) -> Optional[str]:
        """Get and remove the cache_key for a tool_call_id"""
        return self._delegation_cache_key_map.pop(tool_call_id, None)

    def cleanup_delegation_streams(self):
        """Clean up all delegation streams and caches"""
        self._active_delegation_streams.clear()
        self._delegation_result_cache.clear()
        self._delegation_streamed_content.clear()
        self._delegation_cache_key_map.clear()
