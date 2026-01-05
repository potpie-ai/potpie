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
        tool_name = event.result.tool_name or "unknown"
        tool_result = create_tool_result_response(event)

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
        # Track processed nodes by their object id to prevent duplicate processing
        # This can happen when pydantic_ai emits the same node multiple times during parallel tool calls
        processed_node_ids: set = set()

        # Track node counts for debugging duplicate response issues
        node_counts = {
            "model_request": 0,
            "call_tools": 0,
            "end": 0,
            "other": 0,
            "skipped_duplicates": 0,
        }

        async for node in run:
            # Use object id to uniquely identify each node instance
            node_id = id(node)

            # Determine node type for logging
            is_model_request = Agent.is_model_request_node(node)
            is_call_tools = Agent.is_call_tools_node(node)
            is_end = Agent.is_end_node(node)
            node_type = (
                "model_request"
                if is_model_request
                else "call_tools" if is_call_tools else "end" if is_end else "other"
            )

            # Check if we've already processed this node
            if node_id in processed_node_ids:
                node_counts["skipped_duplicates"] += 1
                logger.warning(
                    f"[{context}] Skipping duplicate node: type={node_type}, node_id={node_id}, "
                    f"total_skipped={node_counts['skipped_duplicates']}"
                )
                continue

            # Mark node as processed
            processed_node_ids.add(node_id)
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

            logger.info(
                f"[{context}] Processing node #{sum(node_counts.values()) - node_counts['skipped_duplicates']}: "
                f"type={node_type}, node_id={node_id}, "
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
                async for response in self.process_tool_call_node(
                    node, run.ctx, current_context=current_context
                ):
                    yield response

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

                logger.info(f"[process_tool_call_node] Starting tool call processing")

                async def consume_delegation_queue(
                    tool_call_id: str,
                    input_queue: asyncio.Queue,
                    output_queue: asyncio.Queue,
                ):
                    """Continuously consume from delegation queue and forward to output queue"""
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
                                    await output_queue.put(None)
                                    break
                                await output_queue.put(chunk)
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
                                                await output_queue.put(None)
                                                break
                                            await output_queue.put(chunk)
                                    except asyncio.QueueEmpty:
                                        await output_queue.put(None)
                                    break
                                # Continue waiting
                                continue
                    except asyncio.CancelledError:
                        # Task was cancelled, signal completion
                        await output_queue.put(None)

                async for event in handle_stream:
                    # Yield any chunks from active delegation streams immediately
                    # This ensures real-time streaming even when main stream is idle
                    # Only process streams that haven't been fully drained yet
                    for queue_key in list(output_queues.keys()):
                        if queue_key in drained_streams:
                            continue  # Skip streams that have already been fully drained
                        output_queue = output_queues[queue_key]
                        # Consume all available chunks from output queue
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
                                yield chunk
                            except asyncio.QueueEmpty:
                                break

                    if isinstance(event, FunctionToolCallEvent):
                        tool_call_event_count += 1
                        tool_call_id = event.part.tool_call_id or ""
                        tool_name = event.part.tool_name

                        logger.info(
                            f"[process_tool_call_node] FunctionToolCallEvent #{tool_call_event_count}: "
                            f"tool_name={tool_name}, tool_call_id={tool_call_id[:8]}..."
                        )

                        # Yield the tool call event
                        yield ChatAgentResponse(
                            response="",
                            tool_calls=[create_tool_call_response(event)],
                            citations=[],
                        )

                        # If this is a delegation tool, start streaming the subagent response
                        if (
                            is_delegation_tool(tool_name)
                            and tool_call_id
                            and current_context
                        ):
                            try:
                                # Extract task info from tool call arguments
                                args_dict = event.part.args_as_dict()
                                task_description = args_dict.get("task_description", "")
                                context_str = args_dict.get("context", "")
                                agent_type_str = (
                                    extract_agent_type_from_delegation_tool(tool_name)
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

                                # Start Redis stream consumer task for tool call streaming
                                async def consume_redis_stream_to_queue(
                                    call_id: str,
                                    tool_name: str,
                                    redis_queue: asyncio.Queue,
                                ):
                                    """Consume Redis stream for tool call and put updates in queue"""
                                    try:
                                        async for (
                                            stream_event
                                        ) in self.tool_call_stream_manager.consume_stream(
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

                        logger.info(
                            f"[process_tool_call_node] FunctionToolResultEvent #{tool_result_event_count}: "
                            f"tool_name={tool_name}, tool_call_id={tool_call_id[:8]}..."
                        )

                        # If this was a delegation tool, drain any remaining chunks
                        # Only drain if we haven't already drained this stream
                        if (
                            is_delegation_tool(tool_name)
                            and tool_call_id
                            and tool_call_id not in drained_streams
                        ):
                            # Drain any remaining chunks from output queue
                            if tool_call_id in output_queues:
                                output_queue = output_queues[tool_call_id]
                                # Drain chunks with multiple attempts to catch all
                                for _ in range(10):  # Try up to 10 times
                                    chunks, completed = await self.consume_queue_chunks(
                                        output_queue, timeout=0.1, max_chunks=50
                                    )
                                    for chunk in chunks:
                                        yield chunk
                                    if completed:
                                        # Mark as drained after completion
                                        drained_streams.add(tool_call_id)
                                        break
                                    # Small delay to allow more chunks to arrive
                                    await asyncio.sleep(0.05)

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

                        async for response in self.yield_tool_result_event(event):
                            yield response

                # After all events are processed, drain any remaining chunks from output queues
                # Use output_queues (not active_streams) since that's where chunks are actually stored
                # Only drain streams that haven't been fully drained yet
                for queue_key in list(output_queues.keys()):
                    if queue_key in drained_streams:
                        # Already fully drained, just clean up
                        output_queues.pop(queue_key, None)
                        # Only cleanup active_streams and delegation_manager for non-Redis queues
                        if not queue_key.endswith("_redis"):
                            active_streams.pop(queue_key, None)
                            self.delegation_manager.remove_active_stream(queue_key)
                        continue
                    output_queue = output_queues[queue_key]
                    # Wait longer for final chunks
                    for _ in range(20):  # Try up to 20 times
                        chunks, completed = await self.consume_queue_chunks(
                            output_queue, timeout=0.1, max_chunks=50
                        )
                        for chunk in chunks:
                            yield chunk
                        if completed:
                            drained_streams.add(queue_key)
                            break
                        await asyncio.sleep(0.05)
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

                # Log summary of tool call processing
                logger.info(
                    f"[process_tool_call_node] Completed: "
                    f"tool_calls={tool_call_event_count}, tool_results={tool_result_event_count}, "
                    f"drained_streams={len(drained_streams)}"
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
