"""Delegation manager for handling agent delegation state and functions"""

import asyncio
from typing import Dict, List, Callable, Optional, Any
from pydantic_ai import RunContext

from .utils.delegation_utils import (
    AgentType,
    create_delegation_cache_key,
    format_delegation_error,
    extract_task_result_from_response,
)
from .utils.context_utils import create_project_context_info
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

        # Track active streaming tasks for delegation tools (tool_call_id -> queue)
        self._active_delegation_streams: Dict[str, asyncio.Queue] = {}
        # Cache results from streaming to avoid duplicate execution (task_key -> result)
        self._delegation_result_cache: Dict[str, str] = {}
        # Store streamed content for each delegation (cache_key -> list of chunks)
        self._delegation_streamed_content: Dict[str, List[ChatAgentResponse]] = {}
        # Map tool_call_id to cache_key for retrieving streamed content
        self._delegation_cache_key_map: Dict[str, str] = {}

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

                while waited < max_wait_time:
                    if cache_key in self._delegation_result_cache:
                        result = self._delegation_result_cache.pop(
                            cache_key
                        )  # Remove from cache
                        return result

                    await asyncio.sleep(poll_interval)
                    waited += poll_interval

                # Timeout - this shouldn't happen normally
                logger.error(
                    f"Timeout waiting for delegation result (key={cache_key}). "
                    f"This may indicate the streaming task failed to start."
                )
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
    ):
        """Stream subagent response to a queue for real-time streaming.

        This is the ONLY place where the subagent actually executes.
        The result is cached so delegate_function can return it without re-executing.

        NOTE: Due to pydantic_ai's architecture, streaming during tool execution is limited.
        We collect all chunks and store them for yielding when the tool completes.
        The chunks are stored in _delegation_streamed_content for the tool result handler.
        """
        full_response = ""
        collected_chunks: List[ChatAgentResponse] = []

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

            # Stream the subagent response and collect all chunks
            async for chunk in self.delegation_streamer.stream_subagent_response(
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

            # Cache the result for delegate_function to retrieve
            if full_response:
                # Extract task result section
                summary = extract_task_result_from_response(full_response)
                self._delegation_result_cache[cache_key] = (
                    summary if summary else full_response
                )
            else:
                self._delegation_result_cache[cache_key] = (
                    "## Task Result\n\nNo output from subagent."
                )

        except Exception as e:
            logger.error(f"Error streaming subagent response: {e}", exc_info=True)
            # Put error response in queue
            error_chunk = self.create_error_response(
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
