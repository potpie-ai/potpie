"""Delegation streamer for streaming subagent responses"""

from typing import AsyncGenerator, List, Optional, Any
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    TextPart,
    ModelMessage,
)
from pydantic_ai.usage import UsageLimits

from app.modules.intelligence.agents.chat_agent import ChatAgentResponse
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class DelegationStreamer:
    """Handles streaming of subagent responses"""

    def __init__(
        self,
        handle_stream_error: Any,
        create_error_response: Any,
        process_tool_call_node: Any = None,
    ):
        """Initialize the delegation streamer

        Args:
            handle_stream_error: Function to handle stream errors
            create_error_response: Function to create error responses
            process_tool_call_node: Function to process tool call nodes (optional)
        """
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
                        error_response = self.handle_stream_error(
                            e, f"{agent_type} model request stream"
                        )
                        if error_response:
                            yield error_response
                        continue

                elif Agent.is_call_tools_node(node):
                    # Stream tool calls and results from subagent
                    if self.process_tool_call_node:
                        async for response in self.process_tool_call_node(
                            node, run.ctx
                        ):
                            yield response
                    else:
                        # If no process_tool_call_node provided, skip tool calls
                        # This should not happen in normal operation
                        logger.warning(
                            "process_tool_call_node not provided to DelegationStreamer"
                        )

                elif Agent.is_end_node(node):
                    # Finalize and save reasoning content
                    reasoning_hash = reasoning_manager.finalize_and_save()
                    if reasoning_hash:
                        logger.info(
                            f"Reasoning content saved with hash: {reasoning_hash}"
                        )
                    break

    @staticmethod
    async def collect_agent_streaming_response(
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
                            if isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                full_response += event.delta.content_delta
                                # Accumulate TextPartDelta content for reasoning dump
                                reasoning_manager.append_content(
                                    event.delta.content_delta
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
