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
from app.modules.intelligence.agents.chat_agent import ChatAgentResponse
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
        async for node in run:
            if Agent.is_model_request_node(node):
                # Stream tokens from the model's request
                try:
                    async with node.stream(run.ctx) as request_stream:
                        async for chunk in self.yield_text_stream_events(
                            request_stream, context
                        ):
                            yield chunk
                except Exception as e:
                    error_response = self.handle_stream_error(
                        e, f"{context} model request stream"
                    )
                    if error_response:
                        yield error_response
                    continue

            elif Agent.is_call_tools_node(node):
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
                    for tool_call_id in list(active_streams.keys()):
                        if tool_call_id in output_queues:
                            output_queue = output_queues[tool_call_id]
                            # Consume all available chunks from output queue
                            while True:
                                try:
                                    chunk = output_queue.get_nowait()
                                    if chunk is None:
                                        # Stream completed, clean up
                                        active_streams.pop(tool_call_id, None)
                                        output_queues.pop(tool_call_id, None)
                                        self.delegation_manager.remove_active_stream(
                                            tool_call_id
                                        )
                                        # Cancel and wait for consumer task
                                        if tool_call_id in queue_consumer_tasks:
                                            task = queue_consumer_tasks.pop(
                                                tool_call_id
                                            )
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
                                        break
                                    yield chunk
                                except asyncio.QueueEmpty:
                                    break

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
                                    )
                                )
                                streaming_tasks[tool_call_id] = streaming_task

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
                        tool_call_id = event.result.tool_call_id or ""
                        tool_name = event.result.tool_name or "unknown"

                        # If this was a delegation tool, drain any remaining chunks
                        if is_delegation_tool(tool_name) and tool_call_id:
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

                        async for response in self.yield_tool_result_event(event):
                            yield response

                # After all events are processed, drain any remaining chunks
                for tool_call_id in list(active_streams.keys()):
                    queue = active_streams[tool_call_id]
                    # Wait longer for final chunks
                    for _ in range(20):  # Try up to 20 times
                        chunks, completed = await self.consume_queue_chunks(
                            queue, timeout=0.1, max_chunks=50
                        )
                        for chunk in chunks:
                            yield chunk
                        if completed:
                            break
                        await asyncio.sleep(0.05)
                    active_streams.pop(tool_call_id, None)
                    self.delegation_manager.remove_active_stream(tool_call_id)

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
