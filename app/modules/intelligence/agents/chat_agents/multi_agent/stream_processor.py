"""Stream processor for handling agent run nodes and streaming events"""

import asyncio
import traceback
from typing import AsyncGenerator, List, Optional, Any, Dict, Callable
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    TextPart,
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError
import anyio

from .utils.delegation_utils import (
    is_delegation_tool,
    extract_agent_type_from_delegation_tool,
    create_delegation_cache_key,
)
from .utils.tool_utils import create_tool_call_response, create_tool_result_response
from .utils.tool_call_stream_manager import ToolCallStreamManager
from app.modules.intelligence.agents.chat_agent import (
    ChatAgentResponse,
    ToolCallResponse,
    ToolCallEventType,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tool name constants
TOOL_NAME_SHOW_UPDATED_FILE = "show_updated_file"
TOOL_NAME_SHOW_DIFF = "show_diff"


class StreamProcessor:
    """Processes agent run nodes and handles streaming events"""

    def __init__(
        self,
        delegation_manager: Any,
        create_error_response: Callable[[str], ChatAgentResponse],
    ):
        """Initialize the stream processor

        Args:
            delegation_manager: DelegationManager instance for managing delegation state
            create_error_response: Function to create error responses
        """
        self.delegation_manager = delegation_manager
        self.create_error_response = create_error_response
        self.tool_call_stream_manager = ToolCallStreamManager()

    @staticmethod
    async def yield_text_stream_events(
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

    def handle_stream_error(
        self, error: Exception, context: str = "model request stream"
    ) -> Optional[ChatAgentResponse]:
        """Handle streaming errors and return appropriate response"""
        if isinstance(error, (ModelRetry, AgentRunError, UserError)):
            logger.warning(f"Pydantic-ai error in {context}: {error}")
            return self.create_error_response(
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
                return self.create_error_response(
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
                return self.create_error_response(
                    "*Encountered a parsing error. Skipping this step and continuing...*"
                )
            return self.create_error_response(
                "*An unexpected error occurred. Continuing...*"
            )

    @staticmethod
    async def consume_queue_chunks(
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

    @staticmethod
    async def yield_tool_result_event(
        event: FunctionToolResultEvent,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Yield appropriate response for tool result events"""
        from .utils.delegation_utils import is_delegation_tool

        tool_name = event.result.tool_name or "unknown"
        tool_result = create_tool_result_response(event)
        is_delegation = is_delegation_tool(tool_name)

        # CRITICAL: Ensure delegation results are ALWAYS marked as complete
        if is_delegation:
            if not tool_result.is_complete:
                logger.warning(
                    f"[yield_tool_result_event] Delegation tool result was not marked complete, fixing: "
                    f"tool_name={tool_name}, call_id={event.result.tool_call_id or 'N/A'}"
                )
            tool_result.is_complete = True
            logger.info(
                f"[yield_tool_result_event] Yielding delegation result: tool_name={tool_name}, "
                f"call_id={event.result.tool_call_id or 'N/A'}, is_complete={tool_result.is_complete}, "
                f"event_type={tool_result.event_type}"
            )

        # For show_updated_file and show_diff, append content directly to response
        # instead of going through tool_result_info - these stream directly to user
        if tool_name in (TOOL_NAME_SHOW_UPDATED_FILE, TOOL_NAME_SHOW_DIFF):
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

    async def process_agent_run_nodes(
        self, run: Any, context: str = "agent", current_context: Any = None
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process nodes from an agent run and yield responses

        Args:
            run: Agent run object
            context: Context string for logging
            current_context: Current chat context (optional, needed for delegations)
        """
        # Track processed nodes to prevent duplicate processing
        # IMPORTANT: We keep strong references to node objects, not just their IDs.
        # Using id() alone is unsafe because Python can reuse memory addresses after
        # garbage collection, causing false duplicate detection and missed delegation starts.
        processed_nodes: list = []  # Keep references to prevent GC

        # Track node counts for debugging
        node_counts = {
            "model_request": 0,
            "call_tools": 0,
            "end": 0,
            "other": 0,
            "skipped_duplicates": 0,
        }

        async for node in run:
            # Determine node type for logging
            is_model_request = Agent.is_model_request_node(node)
            is_call_tools = Agent.is_call_tools_node(node)
            is_end = Agent.is_end_node(node)
            node_type = (
                "model_request"
                if is_model_request
                else "call_tools"
                if is_call_tools
                else "end"
                if is_end
                else "other"
            )

            # Check if we've already processed this exact node object
            # Use 'is' for identity comparison (same object in memory)
            is_duplicate = any(node is processed for processed in processed_nodes)
            if is_duplicate:
                node_counts["skipped_duplicates"] += 1
                logger.warning(
                    f"[{context}] Skipping duplicate node: type={node_type}, node_id={id(node)}, "
                    f"total_skipped={node_counts['skipped_duplicates']}"
                )
                continue

            # Keep reference to prevent GC and mark as processed
            processed_nodes.append(node)
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

            logger.info(
                f"[{context}] Processing node #{sum(node_counts.values()) - node_counts['skipped_duplicates']}: "
                f"type={node_type}, node_id={id(node)}, "
                f"counts={{model_request: {node_counts['model_request']}, call_tools: {node_counts['call_tools']}, end: {node_counts['end']}}}"
            )

            if is_model_request:
                # Stream tokens from the model's request
                logger.info(
                    f"[{context}] Starting model request stream (model_request #{node_counts['model_request']})"
                )
                try:
                    async with node.stream(run.ctx) as request_stream:
                        chunk_count = 0
                        async for chunk in self.yield_text_stream_events(
                            request_stream, context
                        ):
                            chunk_count += 1
                            yield chunk
                        logger.info(
                            f"[{context}] Finished model request stream: yielded {chunk_count} chunks"
                        )
                except Exception as e:
                    error_response = self.handle_stream_error(
                        e, f"{context} model request stream"
                    )
                    if error_response:
                        yield error_response
                    continue

            elif is_call_tools:
                # Handle tool calls and results
                context_type = "SUPERVISOR" if current_context else "SUBAGENT"
                logger.info(
                    f"[{context}] Processing call_tools node ({context_type} context), "
                    f"node_id={id(node)}, current_context={current_context is not None}"
                )
                async for response in self.process_tool_call_node(
                    node, run.ctx, current_context=current_context
                ):
                    yield response
                logger.info(
                    f"[{context}] Completed call_tools node ({context_type} context), node_id={id(node)}"
                )

            elif Agent.is_end_node(node):
                # Finalize and save reasoning content
                from app.modules.intelligence.tools.reasoning_manager import (
                    _get_reasoning_manager,
                )

                reasoning_manager = _get_reasoning_manager()
                reasoning_hash = reasoning_manager.finalize_and_save()
                if reasoning_hash:
                    logger.info(f"Reasoning content saved with hash: {reasoning_hash}")
                break

    async def process_tool_call_node(
        self, node: Any, ctx: Any, current_context: Any = None
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Process tool call nodes and yield responses

        Args:
            node: Tool call node from agent run
            ctx: Run context
            current_context: Current chat context (optional, needed for delegations)
        """
        try:
            async with node.stream(ctx) as handle_stream:
                # Track active delegation streams for this tool call node
                active_streams: Dict[str, asyncio.Queue] = {}
                # Track streaming tasks to ensure they complete
                streaming_tasks: Dict[str, asyncio.Task] = {}
                # Track queue consumer tasks for real-time streaming
                queue_consumer_tasks: Dict[str, asyncio.Task] = {}
                # Track output queues for each consumer task
                output_queues: Dict[str, asyncio.Queue] = {}
                # Track which streams have been fully drained (to prevent duplicate draining)
                drained_streams: set = set()
                # Track Redis stream consumer tasks for tool call streaming
                redis_stream_tasks: Dict[str, asyncio.Task] = {}

                # Track event counts for debugging
                tool_call_event_count = 0
                tool_result_event_count = 0

                context_type = "SUPERVISOR" if current_context else "SUBAGENT"
                logger.info(
                    f"[process_tool_call_node] Starting tool call processing ({context_type} context)"
                )

                async def consume_delegation_queue(
                    tool_call_id: str,
                    input_queue: asyncio.Queue,
                    output_queue: asyncio.Queue,
                ):
                    """Continuously consume from delegation queue and forward to output queue"""
                    logger.info(
                        f"[consume_delegation_queue] Starting consumer for tool_call_id={tool_call_id[:8]}..."
                    )
                    chunks_consumed = 0
                    try:
                        while True:
                            try:
                                # Wait for chunk with a short timeout to allow cancellation
                                chunk = await asyncio.wait_for(
                                    input_queue.get(), timeout=0.1
                                )
                                if (
                                    chunk is None
                                ):  # Sentinel value indicating completion
                                    try:
                                        output_queue.put_nowait(None)
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            f"[consume_delegation_queue] Output queue full for {tool_call_id}"
                                        )
                                    break
                                # Use put_nowait to avoid blocking - queue should be unbounded
                                try:
                                    output_queue.put_nowait(chunk)
                                    chunks_consumed += 1
                                    if chunks_consumed % 10 == 0:
                                        logger.debug(
                                            f"[consume_delegation_queue] Consumed {chunks_consumed} chunks "
                                            f"for tool_call_id={tool_call_id[:8]}..."
                                        )
                                except asyncio.QueueFull:
                                    logger.warning(
                                        f"[consume_delegation_queue] Output queue full, dropping chunk for {tool_call_id}"
                                    )
                            except asyncio.TimeoutError:
                                # Check if the streaming task is done
                                task = streaming_tasks.get(tool_call_id)
                                if task and task.done():
                                    # Task completed, check for final None
                                    try:
                                        # Try to get any remaining items
                                        while True:
                                            chunk = input_queue.get_nowait()
                                            if chunk is None:
                                                try:
                                                    output_queue.put_nowait(None)
                                                except asyncio.QueueFull:
                                                    pass
                                                break
                                            try:
                                                output_queue.put_nowait(chunk)
                                            except asyncio.QueueFull:
                                                logger.warning(
                                                    f"[consume_delegation_queue] Output queue full, dropping chunk for {tool_call_id}"
                                                )
                                    except asyncio.QueueEmpty:
                                        try:
                                            output_queue.put_nowait(None)
                                        except asyncio.QueueFull:
                                            pass
                                    break
                                # Continue waiting
                                continue
                    except asyncio.CancelledError:
                        # Task was cancelled, signal completion
                        logger.info(
                            f"[consume_delegation_queue] Consumer cancelled for tool_call_id={tool_call_id[:8]}..., "
                            f"consumed {chunks_consumed} chunks total"
                        )
                        try:
                            output_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                    except Exception as e:
                        logger.error(
                            f"[consume_delegation_queue] Error in consumer for tool_call_id={tool_call_id[:8]}...: {e}",
                            exc_info=True,
                        )
                        try:
                            output_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass

                async for event in handle_stream:
                    # Yield any chunks from active delegation streams immediately
                    # This ensures real-time streaming even when main stream is idle
                    # Only process streams that haven't been fully drained yet
                    for queue_key in list(output_queues.keys()):
                        if queue_key in drained_streams:
                            continue  # Skip streams that have already been fully drained
                        output_queue = output_queues[queue_key]
                        # Consume all available chunks from output queue
                        chunks_yielded = 0
                        while True:
                            try:
                                chunk = output_queue.get_nowait()
                                if chunk is None:
                                    # Stream completed, mark as drained and clean up
                                    drained_streams.add(queue_key)
                                    output_queues.pop(queue_key, None)
                                    # Only cleanup active_streams and tasks for non-Redis queues
                                    if not queue_key.endswith("_redis"):
                                        active_streams.pop(queue_key, None)
                                        self.delegation_manager.remove_active_stream(
                                            queue_key
                                        )
                                        # Cancel and wait for consumer task
                                        if queue_key in queue_consumer_tasks:
                                            task = queue_consumer_tasks.pop(queue_key)
                                            if not task.done():
                                                task.cancel()
                                                try:
                                                    await task
                                                except asyncio.CancelledError:
                                                    pass
                                        # Cancel the streaming task if it's still running
                                        if queue_key in streaming_tasks:
                                            task = streaming_tasks.pop(queue_key)
                                            if not task.done():
                                                task.cancel()
                                                try:
                                                    await task
                                                except asyncio.CancelledError:
                                                    pass
                                    else:
                                        # Clean up Redis stream task
                                        actual_call_id = queue_key.replace("_redis", "")
                                        if actual_call_id in redis_stream_tasks:
                                            task = redis_stream_tasks.pop(
                                                actual_call_id
                                            )
                                            if not task.done():
                                                task.cancel()
                                                try:
                                                    await task
                                                except asyncio.CancelledError:
                                                    pass
                                    break
                                chunks_yielded += 1
                                # Subagent chunks should only contain text (tool calls are filtered out)
                                # Log text content for debugging
                                if chunk.response:
                                    logger.debug(
                                        f"[process_tool_call_node] Yielding subagent text chunk "
                                        f"(queue_key={queue_key}, length={len(chunk.response)})"
                                    )
                                yield chunk
                            except asyncio.QueueEmpty:
                                if chunks_yielded > 0:
                                    logger.debug(
                                        f"[process_tool_call_node] Yielded {chunks_yielded} chunks from queue {queue_key}"
                                    )
                                break

                    if isinstance(event, FunctionToolCallEvent):
                        tool_call_event_count += 1
                        tool_call_id = event.part.tool_call_id or ""
                        tool_name = event.part.tool_name

                        # Log context: supervisor or subagent
                        context_type = "SUPERVISOR" if current_context else "SUBAGENT"

                        # Check if this is a delegation tool
                        is_delegation = is_delegation_tool(tool_name)

                        logger.info(
                            f"[process_tool_call_node] FunctionToolCallEvent #{tool_call_event_count} ({context_type}): "
                            f"tool_name={tool_name}, tool_call_id={tool_call_id[:8]}..., "
                            f"is_delegation={is_delegation}"
                        )

                        # CRITICAL: Log when supervisor calls a delegation tool
                        if current_context and is_delegation:
                            logger.info(
                                f"[process_tool_call_node] 🎯 SUPERVISOR DELEGATION TOOL CALL: "
                                f"tool_name={tool_name}, call_id={tool_call_id[:8]}..., "
                                f"context_type={context_type}"
                            )

                        # Yield the tool call event
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[create_tool_call_response(event)],
                            citations=[],
                        )

                        # If this is a delegation tool, start streaming the subagent response
                        logger.info(
                            f"[process_tool_call_node] Checking delegation: tool_name={tool_name}, "
                            f"is_delegation_tool={is_delegation}, tool_call_id={tool_call_id[:8] if tool_call_id else 'None'}..., "
                            f"has_current_context={current_context is not None}"
                        )
                        if is_delegation and tool_call_id and current_context:
                            try:
                                # Extract task info from tool call arguments
                                args_dict = event.part.args_as_dict()
                                task_description = args_dict.get("task_description", "")
                                context_str = args_dict.get("context", "")
                                agent_type_str = (
                                    extract_agent_type_from_delegation_tool(tool_name)
                                )
                                logger.info(
                                    f"[process_tool_call_node] Delegation tool detected: tool_name={tool_name}, "
                                    f"agent_type={agent_type_str}, tool_call_id={tool_call_id[:8]}..."
                                )

                                # Create cache key for coordination with delegate_function
                                cache_key = create_delegation_cache_key(
                                    task_description, context_str
                                )

                                # Store the cache_key -> tool_call_id mapping for later retrieval
                                self.delegation_manager.map_cache_key(
                                    tool_call_id, cache_key
                                )

                                # Create queues for streaming chunks
                                # input_queue: receives chunks from subagent
                                # output_queue: forwards chunks to main stream
                                input_queue: asyncio.Queue = asyncio.Queue()
                                output_queue: asyncio.Queue = asyncio.Queue()
                                active_streams[tool_call_id] = input_queue
                                output_queues[tool_call_id] = output_queue
                                self.delegation_manager.register_active_stream(
                                    tool_call_id, input_queue
                                )

                                # Start streaming the subagent response in the background
                                # This is the ONLY place the subagent executes - it will cache the result
                                streaming_task = asyncio.create_task(
                                    self.delegation_manager.stream_subagent_to_queue(
                                        agent_type_str,
                                        task_description,
                                        context_str,
                                        input_queue,
                                        cache_key,
                                        current_context,
                                        call_id=tool_call_id,  # Pass call_id for Redis streaming
                                    )
                                )
                                streaming_tasks[tool_call_id] = streaming_task

                                # CRITICAL: Register the streaming task with the cache_key so delegate_function
                                # can check if it's running and detect failures
                                self.delegation_manager._active_streaming_tasks[
                                    cache_key
                                ] = streaming_task
                                logger.info(
                                    f"[process_tool_call_node] Registered streaming task for cache_key={cache_key}, "
                                    f"tool_call_id={tool_call_id[:8]}..."
                                )

                                # Start Redis stream consumer task for tool call streaming
                                async def consume_redis_stream_to_queue(
                                    call_id: str,
                                    tool_name: str,
                                    redis_queue: asyncio.Queue,
                                ):
                                    """Consume Redis stream for tool call and put updates in queue"""
                                    try:
                                        async for stream_event in self.tool_call_stream_manager.consume_stream(
                                            call_id
                                        ):
                                            if (
                                                stream_event.get("type")
                                                == "tool_call_stream_part"
                                            ):
                                                stream_part = stream_event.get(
                                                    "stream_part", ""
                                                )
                                                is_complete = (
                                                    stream_event.get(
                                                        "is_complete", "false"
                                                    )
                                                    == "true"
                                                )
                                                tool_response = stream_event.get(
                                                    "tool_response", ""
                                                )
                                                tool_call_details = stream_event.get(
                                                    "tool_call_details", {}
                                                )

                                                # Create tool call response with stream_part
                                                stream_tool_response = ToolCallResponse(
                                                    call_id=call_id,
                                                    event_type=(
                                                        ToolCallEventType.DELEGATION_RESULT
                                                        if is_delegation_tool(tool_name)
                                                        else ToolCallEventType.RESULT
                                                    ),
                                                    tool_name=tool_name,
                                                    tool_response=tool_response
                                                    or stream_part,
                                                    tool_call_details=tool_call_details,
                                                    stream_part=stream_part,
                                                    is_complete=is_complete,
                                                )

                                                await redis_queue.put(
                                                    ChatAgentResponse(
                                                        response="",
                                                        tool_calls=[
                                                            stream_tool_response
                                                        ],
                                                        citations=[],
                                                    )
                                                )

                                                if is_complete:
                                                    await redis_queue.put(None)
                                                    break

                                            elif (
                                                stream_event.get("type")
                                                == "tool_call_stream_end"
                                            ):
                                                await redis_queue.put(None)
                                                break

                                    except Exception as e:
                                        logger.warning(
                                            f"Error consuming Redis stream for call_id {call_id}: {e}"
                                        )
                                        await redis_queue.put(None)

                                # Create a queue for Redis stream updates
                                redis_stream_queue: asyncio.Queue = asyncio.Queue()
                                redis_stream_task = asyncio.create_task(
                                    consume_redis_stream_to_queue(
                                        tool_call_id, tool_name, redis_stream_queue
                                    )
                                )
                                redis_stream_tasks[tool_call_id] = redis_stream_task
                                # Store the queue for consumption in the main loop
                                output_queues[f"{tool_call_id}_redis"] = (
                                    redis_stream_queue
                                )

                                # Start a background task to continuously consume from input_queue
                                # and forward to output_queue for real-time streaming
                                consumer_task = asyncio.create_task(
                                    consume_delegation_queue(
                                        tool_call_id, input_queue, output_queue
                                    )
                                )
                                queue_consumer_tasks[tool_call_id] = consumer_task
                                logger.info(
                                    f"[process_tool_call_node] Started queue consumer task for "
                                    f"tool_call_id={tool_call_id[:8]}..."
                                )

                                # Yield a visual indicator that subagent is starting
                                yield ChatAgentResponse(
                                    response="\n\n---\n🤖 **Subagent Starting...**\n\n",
                                    tool_calls=[],
                                    citations=[],
                                )

                                # Immediately try to consume any early chunks from output queue
                                chunks, completed = await self.consume_queue_chunks(
                                    output_queue, timeout=0.01, max_chunks=5
                                )
                                for chunk in chunks:
                                    yield chunk
                                if completed:
                                    # Clean up if stream completed immediately
                                    active_streams.pop(tool_call_id, None)
                                    output_queues.pop(tool_call_id, None)
                                    self.delegation_manager.remove_active_stream(
                                        tool_call_id
                                    )
                                    if tool_call_id in queue_consumer_tasks:
                                        task = queue_consumer_tasks.pop(tool_call_id)
                                        if not task.done():
                                            task.cancel()
                                    if tool_call_id in streaming_tasks:
                                        task = streaming_tasks.pop(tool_call_id)
                                        if not task.done():
                                            task.cancel()

                            except Exception as e:
                                logger.warning(
                                    f"Error setting up subagent streaming for {tool_name}: {e}"
                                )
                                # Clean up the queues and tasks if there was an error
                                if tool_call_id in active_streams:
                                    del active_streams[tool_call_id]
                                if tool_call_id in output_queues:
                                    del output_queues[tool_call_id]
                                self.delegation_manager.remove_active_stream(
                                    tool_call_id
                                )
                                if tool_call_id in queue_consumer_tasks:
                                    task = queue_consumer_tasks.pop(tool_call_id)
                                    if not task.done():
                                        task.cancel()
                                if tool_call_id in streaming_tasks:
                                    task = streaming_tasks.pop(tool_call_id)
                                    if not task.done():
                                        task.cancel()

                    if isinstance(event, FunctionToolResultEvent):
                        tool_result_event_count += 1
                        tool_call_id = event.result.tool_call_id or ""
                        tool_name = event.result.tool_name or "unknown"
                        is_delegation = is_delegation_tool(tool_name)

                        # Log context: supervisor or subagent
                        context_type = "SUPERVISOR" if current_context else "SUBAGENT"

                        logger.info(
                            f"[process_tool_call_node] FunctionToolResultEvent #{tool_result_event_count} ({context_type}): "
                            f"tool_name={tool_name}, tool_call_id={tool_call_id[:8]}..., "
                            f"is_delegation={is_delegation}, content_length={len(str(event.result.content)) if event.result.content else 0}"
                        )

                        # CRITICAL: Log when we see a delegation tool result from supervisor
                        if is_delegation and current_context:
                            logger.info(
                                f"[process_tool_call_node] ⚠️ CRITICAL: Supervisor delegation tool result detected! "
                                f"tool_name={tool_name}, call_id={tool_call_id[:8]}..., "
                                f"content_preview={str(event.result.content)[:200] if event.result.content else 'None'}..."
                            )

                        # If this was a delegation tool, drain any remaining chunks
                        # Only drain if we haven't already drained this stream
                        if (
                            is_delegation_tool(tool_name)
                            and tool_call_id
                            and tool_call_id not in drained_streams
                        ):
                            logger.info(
                                f"[process_tool_call_node] Draining delegation stream for tool_call_id={tool_call_id[:8]}... "
                                f"(tool_name={tool_name}, context_type={context_type})"
                            )
                            drain_start_time = asyncio.get_event_loop().time()
                            # Drain any remaining chunks from output queue
                            if tool_call_id in output_queues:
                                output_queue = output_queues[tool_call_id]
                                total_drained_chunks = 0
                                # Drain chunks with multiple attempts to catch all
                                for attempt in range(10):  # Try up to 10 times
                                    logger.debug(
                                        f"[process_tool_call_node] Drain attempt {attempt + 1}/10 for "
                                        f"tool_call_id={tool_call_id[:8]}..."
                                    )
                                    chunks, completed = await self.consume_queue_chunks(
                                        output_queue, timeout=0.1, max_chunks=50
                                    )
                                    total_drained_chunks += len(chunks)
                                    for chunk in chunks:
                                        yield chunk
                                    if completed:
                                        # Mark as drained after completion
                                        drained_streams.add(tool_call_id)
                                        drain_elapsed = (
                                            asyncio.get_event_loop().time()
                                            - drain_start_time
                                        )
                                        logger.info(
                                            f"[process_tool_call_node] Stream drained successfully: "
                                            f"tool_call_id={tool_call_id[:8]}..., chunks={total_drained_chunks}, "
                                            f"attempts={attempt + 1}, elapsed={drain_elapsed:.2f}s"
                                        )
                                        break
                                    # Small delay to allow more chunks to arrive
                                    await asyncio.sleep(0.05)
                                else:
                                    drain_elapsed = (
                                        asyncio.get_event_loop().time()
                                        - drain_start_time
                                    )
                                    logger.warning(
                                        f"[process_tool_call_node] ⚠️ Stream drain incomplete after 10 attempts: "
                                        f"tool_call_id={tool_call_id[:8]}..., chunks={total_drained_chunks}, "
                                        f"elapsed={drain_elapsed:.2f}s. Stream may still be active."
                                    )
                            else:
                                logger.warning(
                                    f"[process_tool_call_node] No output queue found for tool_call_id={tool_call_id[:8]}... "
                                    f"when trying to drain"
                                )

                            # Add a visual separator after subagent output
                            yield ChatAgentResponse(
                                response="\n\n---\n✅ **Subagent Complete**\n\n",
                                tool_calls=[],
                                citations=[],
                            )

                            # Clean up streams and tasks
                            active_streams.pop(tool_call_id, None)
                            output_queues.pop(tool_call_id, None)
                            output_queues.pop(
                                f"{tool_call_id}_redis", None
                            )  # Clean up Redis queue
                            self.delegation_manager.remove_active_stream(tool_call_id)

                            # Cancel and wait for consumer task
                            if tool_call_id in queue_consumer_tasks:
                                task = queue_consumer_tasks.pop(tool_call_id)
                                if not task.done():
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

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

                            # Cancel Redis stream task if it's still running
                            if tool_call_id in redis_stream_tasks:
                                task = redis_stream_tasks.pop(tool_call_id)
                                if not task.done():
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

                        # Yield the final tool result - this signals completion to the frontend
                        # CRITICAL: For delegation tools, this MUST be yielded with is_complete=True
                        async for response in self.yield_tool_result_event(event):
                            tool_result = (
                                response.tool_calls[0] if response.tool_calls else None
                            )
                            is_complete = (
                                tool_result.is_complete if tool_result else False
                            )
                            logger.info(
                                f"[process_tool_call_node] Yielding tool result for {tool_name} "
                                f"(call_id={tool_call_id[:8]}...), is_delegation={is_delegation}, "
                                f"is_complete={is_complete}, event_type={tool_result.event_type if tool_result else 'N/A'}"
                            )
                            if is_delegation and not is_complete:
                                logger.error(
                                    f"[process_tool_call_node] CRITICAL: Delegation tool result missing is_complete=True! "
                                    f"tool_name={tool_name}, call_id={tool_call_id[:8]}..."
                                )
                            yield response

                # After all events are processed, drain any remaining chunks from output queues
                # Use output_queues (not active_streams) since that's where chunks are actually stored
                # Only drain streams that haven't been fully drained yet
                logger.info(
                    f"[process_tool_call_node] Starting final drain of {len(output_queues)} output queues "
                    f"(context_type={context_type}, drained_streams={len(drained_streams)})"
                )
                for queue_key in list(output_queues.keys()):
                    if queue_key in drained_streams:
                        # Already fully drained, just clean up
                        logger.debug(
                            f"[process_tool_call_node] Skipping already-drained queue: {queue_key}"
                        )
                        output_queues.pop(queue_key, None)
                        # Only cleanup active_streams and delegation_manager for non-Redis queues
                        if not queue_key.endswith("_redis"):
                            active_streams.pop(queue_key, None)
                            self.delegation_manager.remove_active_stream(queue_key)
                        continue

                    logger.info(
                        f"[process_tool_call_node] Draining final chunks from queue: {queue_key}"
                    )
                    output_queue = output_queues[queue_key]
                    final_drain_start = asyncio.get_event_loop().time()
                    total_final_chunks = 0
                    # Wait longer for final chunks
                    for attempt in range(20):  # Try up to 20 times
                        chunks, completed = await self.consume_queue_chunks(
                            output_queue, timeout=0.1, max_chunks=50
                        )
                        total_final_chunks += len(chunks)
                        for chunk in chunks:
                            yield chunk
                        if completed:
                            drained_streams.add(queue_key)
                            final_drain_elapsed = (
                                asyncio.get_event_loop().time() - final_drain_start
                            )
                            logger.info(
                                f"[process_tool_call_node] Final drain completed for {queue_key}: "
                                f"chunks={total_final_chunks}, attempts={attempt + 1}, "
                                f"elapsed={final_drain_elapsed:.2f}s"
                            )
                            break
                        await asyncio.sleep(0.05)
                    else:
                        final_drain_elapsed = (
                            asyncio.get_event_loop().time() - final_drain_start
                        )
                        logger.warning(
                            f"[process_tool_call_node] ⚠️ Final drain incomplete for {queue_key} after 20 attempts: "
                            f"chunks={total_final_chunks}, elapsed={final_drain_elapsed:.2f}s"
                        )
                    output_queues.pop(queue_key, None)
                    # Only cleanup active_streams and delegation_manager for non-Redis queues
                    if not queue_key.endswith("_redis"):
                        active_streams.pop(queue_key, None)
                        self.delegation_manager.remove_active_stream(queue_key)
                    else:
                        # Clean up Redis stream task for Redis queues
                        actual_call_id = queue_key.replace("_redis", "")
                        if actual_call_id in redis_stream_tasks:
                            task = redis_stream_tasks.pop(actual_call_id)
                            if not task.done():
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass

                # Cancel any remaining streaming tasks
                for tool_call_id, task in list(streaming_tasks.items()):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    streaming_tasks.pop(tool_call_id, None)

                # CRITICAL: For supervisor context, wait for all delegation cached results to be available
                # before closing the stream. This ensures tool result events are generated and processed.
                # We need to wait for the cached results, not just the streaming tasks, because
                # delegate_function waits for the cache, and the tool result event is generated when delegate_function returns.
                if current_context:
                    # Get all cache keys for delegation tool calls in this node
                    delegation_cache_keys = []
                    for tool_call_id in list(streaming_tasks.keys()):
                        cache_key = self.delegation_manager.get_cache_key(tool_call_id)
                        if cache_key:
                            delegation_cache_keys.append((tool_call_id, cache_key))

                    if delegation_cache_keys:
                        logger.info(
                            f"[process_tool_call_node] Waiting for {len(delegation_cache_keys)} delegation cached results "
                            f"before closing stream (SUPERVISOR context)"
                        )
                        # Wait for all cached results to be available
                        # This ensures delegate_function can return and tool result events are generated
                        max_wait = 300  # 5 minutes max
                        wait_interval = 0.5  # 500ms
                        waited = 0
                        pending_keys = set(
                            cache_key for _, cache_key in delegation_cache_keys
                        )
                        wait_start_time = asyncio.get_event_loop().time()

                        while pending_keys and waited < max_wait:
                            # Check which cache keys have results
                            for tool_call_id, cache_key in delegation_cache_keys:
                                if cache_key in pending_keys:
                                    # Check if result is cached
                                    result = (
                                        self.delegation_manager.get_delegation_result(
                                            cache_key
                                        )
                                    )
                                    if result:
                                        pending_keys.discard(cache_key)
                                        logger.info(
                                            f"[process_tool_call_node] Cache result found for tool_call_id={tool_call_id[:8]}..., "
                                            f"cache_key={cache_key}, result_length={len(result)} chars"
                                        )

                            if not pending_keys:
                                wait_elapsed = (
                                    asyncio.get_event_loop().time() - wait_start_time
                                )
                                logger.info(
                                    f"[process_tool_call_node] All delegation cached results available, "
                                    f"stream can close safely (waited {int(waited)}s, elapsed={wait_elapsed:.2f}s)"
                                )
                                break

                            await asyncio.sleep(wait_interval)
                            waited += wait_interval

                            if int(waited) % 5 == 0:
                                wait_elapsed = (
                                    asyncio.get_event_loop().time() - wait_start_time
                                )
                                # Check task status for pending keys
                                task_statuses = {}
                                for tool_call_id, cache_key in delegation_cache_keys:
                                    if cache_key in pending_keys:
                                        task = streaming_tasks.get(tool_call_id)
                                        if task:
                                            task_statuses[cache_key] = (
                                                f"done={task.done()}, cancelled={task.cancelled()}"
                                            )
                                        else:
                                            task_statuses[cache_key] = "task_not_found"
                                logger.warning(
                                    f"[process_tool_call_node] ⚠️ Still waiting for {len(pending_keys)} cached results "
                                    f"(waited {int(waited)}s, elapsed={wait_elapsed:.2f}s). "
                                    f"Task statuses: {task_statuses}"
                                )

                        if pending_keys:
                            wait_elapsed = (
                                asyncio.get_event_loop().time() - wait_start_time
                            )
                            logger.error(
                                f"[process_tool_call_node] ⚠️ TIMEOUT waiting for {len(pending_keys)} cached results "
                                f"after {int(waited)}s (elapsed={wait_elapsed:.2f}s). "
                                f"Some tool result events may be lost! Pending keys: {list(pending_keys)}"
                            )

                # Log summary of tool call processing
                context_type = "SUPERVISOR" if current_context else "SUBAGENT"
                logger.info(
                    f"[process_tool_call_node] Stream exhausted - Completed ({context_type}): "
                    f"tool_calls={tool_call_event_count}, tool_results={tool_result_event_count}, "
                    f"drained_streams={len(drained_streams)}, remaining_streaming_tasks={len(streaming_tasks)}"
                )

                # CRITICAL: If this was a supervisor context and we processed delegation tool calls
                # but no delegation tool results, log a warning
                if current_context and tool_call_event_count > 0:
                    if tool_call_event_count > tool_result_event_count:
                        logger.warning(
                            f"[process_tool_call_node] ⚠️ SUPERVISOR processed {tool_call_event_count} tool calls "
                            f"but only {tool_result_event_count} tool results. Missing {tool_call_event_count - tool_result_event_count} tool result events!"
                        )

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
                yield self.create_error_response(
                    "*Encountered a message history error. This may require starting a new conversation.*"
                )
            else:
                logger.warning(
                    f"Pydantic-ai error in tool call stream: {pydantic_error}"
                )
                yield self.create_error_response(
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
                yield self.create_error_response(
                    "*Encountered a message history error. This may require starting a new conversation.*"
                )
            else:
                logger.error(f"Unexpected error in tool call stream: {e}")
                yield self.create_error_response(
                    "*An unexpected error occurred during tool execution. Continuing...*"
                )
