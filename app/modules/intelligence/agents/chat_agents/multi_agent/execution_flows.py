"""Execution flows for different agent execution modes"""

import traceback
from typing import AsyncGenerator, Any, Optional
import anyio
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import UsageLimits

from .utils.message_history_utils import (
    validate_and_fix_message_history,
    prepare_multimodal_message_history,
)
from .utils.multimodal_utils import create_multimodal_user_content
from app.modules.intelligence.agents.chat_agent import ChatContext, ChatAgentResponse
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def init_managers(conversation_id: Optional[str] = None):
    """Initialize managers for agent run.

    This initializes all tool managers for the current agent execution context.
    For code changes, this loads any existing changes from Redis for the conversation,
    allowing changes to persist across messages in the same conversation.

    Args:
        conversation_id: The conversation ID for persisting state (e.g., code changes) across messages.
    """
    from app.modules.intelligence.tools.todo_management_tool import (
        _reset_todo_manager,
    )
    from app.modules.intelligence.tools.code_changes_manager import (
        _init_code_changes_manager,
    )
    from app.modules.intelligence.tools.requirement_verification_tool import (
        _reset_requirement_manager,
    )

    _reset_todo_manager()
    _init_code_changes_manager(conversation_id)
    _reset_requirement_manager()
    logger.info(
        f"🔄 Initialized managers for agent run (conversation_id={conversation_id})"
    )


# Backward compatibility alias
reset_managers = init_managers


class StandardExecutionFlow:
    """Standard text-only non-streaming execution flow"""

    def __init__(
        self,
        create_supervisor_agent: Any,
        history_processor: Any,
    ):
        """Initialize the standard execution flow

        Args:
            create_supervisor_agent: Function to create supervisor agent
            history_processor: History processor instance
        """
        self.create_supervisor_agent = create_supervisor_agent
        self.history_processor = history_processor

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Standard text-only multi-agent execution"""
        try:
            # Prepare message history
            message_history = await prepare_multimodal_message_history(
                ctx, self.history_processor
            )

            # Create and run supervisor agent
            supervisor_agent = self.create_supervisor_agent(ctx)

            # Validate message history before sending to model (single validation)
            message_history = validate_and_fix_message_history(message_history)

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
            logger.error(
                f"Error in standard multi-agent run method: {str(e)}", exc_info=True
            )
            return ChatAgentResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                tool_calls=[],
                citations=[],
            )


class MultimodalExecutionFlow:
    """Multimodal non-streaming execution flow"""

    def __init__(
        self,
        create_supervisor_agent: Any,
        history_processor: Any,
        standard_flow: StandardExecutionFlow,
    ):
        """Initialize the multimodal execution flow

        Args:
            create_supervisor_agent: Function to create supervisor agent
            history_processor: History processor instance
            standard_flow: StandardExecutionFlow instance for fallback
        """
        self.create_supervisor_agent = create_supervisor_agent
        self.history_processor = history_processor
        self.standard_flow = standard_flow

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Multimodal multi-agent execution using PydanticAI's native multimodal capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await prepare_multimodal_message_history(
                ctx, self.history_processor
            )

            # Create and run supervisor agent
            supervisor_agent = self.create_supervisor_agent(ctx)

            resp = await supervisor_agent.run(
                user_prompt=multimodal_content,
                message_history=message_history,
            )

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
            return await self.standard_flow.run(ctx)


class StreamingExecutionFlow:
    """Standard text-only streaming execution flow"""

    def __init__(
        self,
        create_supervisor_agent: Any,
        history_processor: Any,
        stream_processor: Any,
        current_supervisor_run_ref: Any,
    ):
        """Initialize the streaming execution flow

        Args:
            create_supervisor_agent: Function to create supervisor agent
            history_processor: History processor instance
            stream_processor: StreamProcessor instance
            current_supervisor_run_ref: Reference to store current supervisor run
        """
        self.create_supervisor_agent = create_supervisor_agent
        self.history_processor = history_processor
        self.stream_processor = stream_processor
        self.current_supervisor_run_ref = current_supervisor_run_ref

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Standard multi-agent streaming execution with MCP server support"""
        # Create supervisor agent directly
        supervisor_agent = self.create_supervisor_agent(ctx)

        try:
            # Try to initialize MCP servers with timeout handling
            try:
                # Use prepare_multimodal_message_history to get compressed history if available
                message_history = await prepare_multimodal_message_history(
                    ctx, self.history_processor
                )
                message_history = validate_and_fix_message_history(message_history)

                async with supervisor_agent.run_mcp_servers():
                    async with supervisor_agent.iter(
                        user_prompt=ctx.query,
                        message_history=message_history,
                        usage_limits=UsageLimits(
                            request_limit=None
                        ),  # No request limit for long-running tasks
                    ) as run:
                        # Store the supervisor run so delegation functions can access its message history
                        self.current_supervisor_run_ref["run"] = run
                        try:
                            async for (
                                response
                            ) in self.stream_processor.process_agent_run_nodes(
                                run, "multi-agent", current_context=ctx
                            ):
                                yield response
                        finally:
                            # Clear the reference when done
                            self.current_supervisor_run_ref["run"] = None

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
                message_history = await prepare_multimodal_message_history(
                    ctx, self.history_processor
                )
                message_history = validate_and_fix_message_history(message_history)

                async with supervisor_agent.iter(
                    user_prompt=ctx.query,
                    message_history=message_history,
                    usage_limits=UsageLimits(
                        request_limit=None
                    ),  # No request limit for long-running tasks
                ) as run:
                    # Store the supervisor run so delegation functions can access its message history
                    self.current_supervisor_run_ref["run"] = run
                    try:
                        async for (
                            response
                        ) in self.stream_processor.process_agent_run_nodes(
                            run, "multi-agent", current_context=ctx
                        ):
                            yield response
                    finally:
                        # Clear the reference when done
                        self.current_supervisor_run_ref["run"] = None

        except Exception as e:
            error_str = str(e)
            # Check if this is a tool retry error from pydantic-ai
            if (
                "exceeded max retries" in error_str.lower()
                and "tool" in error_str.lower()
            ):
                # Extract tool name if possible
                tool_name = "unknown"
                if "'" in error_str:
                    parts = error_str.split("'")
                    if len(parts) >= 2:
                        tool_name = parts[1]

                logger.warning(
                    f"Tool '{tool_name}' exceeded max retries in multi-agent stream. "
                    f"This usually indicates the tool is failing repeatedly. Error: {error_str}",
                    exc_info=True,
                )
                yield ChatAgentResponse(
                    response=f"\n\n*An error occurred while executing tool '{tool_name}'. The tool failed after retries. Please try a different approach.*\n\n",
                    tool_calls=[],
                    citations=[],
                )
            else:
                logger.error(
                    f"Error in standard multi-agent stream: {error_str}", exc_info=True
                )
                yield ChatAgentResponse(
                    response="\n\n*An error occurred during multi-agent streaming*\n\n",
                    tool_calls=[],
                    citations=[],
                )


class MultimodalStreamingExecutionFlow:
    """Multimodal streaming execution flow"""

    def __init__(
        self,
        create_supervisor_agent: Any,
        history_processor: Any,
        stream_processor: Any,
        current_supervisor_run_ref: Any,
        standard_streaming_flow: StreamingExecutionFlow,
    ):
        """Initialize the multimodal streaming execution flow

        Args:
            create_supervisor_agent: Function to create supervisor agent
            history_processor: History processor instance
            stream_processor: StreamProcessor instance
            current_supervisor_run_ref: Reference to store current supervisor run
            standard_streaming_flow: StreamingExecutionFlow instance for fallback
        """
        self.create_supervisor_agent = create_supervisor_agent
        self.history_processor = history_processor
        self.stream_processor = stream_processor
        self.current_supervisor_run_ref = current_supervisor_run_ref
        self.standard_streaming_flow = standard_streaming_flow

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multimodal multi-agent response using PydanticAI's native capabilities"""
        try:
            # Create multimodal user content with images
            multimodal_content = create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await prepare_multimodal_message_history(
                ctx, self.history_processor
            )

            # Validate message history before sending to model
            message_history = validate_and_fix_message_history(message_history)

            # Create supervisor agent
            supervisor_agent = self.create_supervisor_agent(ctx)

            # Stream the response
            async with supervisor_agent.iter(
                user_prompt=multimodal_content,
                message_history=message_history,
                usage_limits=UsageLimits(
                    request_limit=None
                ),  # No request limit for long-running tasks
            ) as run:
                # Store the supervisor run so delegation functions can access its message history
                self.current_supervisor_run_ref["run"] = run
                try:
                    async for response in self.stream_processor.process_agent_run_nodes(
                        run, "multimodal multi-agent", current_context=ctx
                    ):
                        yield response
                finally:
                    # Note: For streaming runs, compressed messages are handled by history processor
                    # Clear the reference when done
                    self.current_supervisor_run_ref["run"] = None

        except Exception as e:
            logger.error(
                f"Error in multimodal multi-agent stream: {str(e)}", exc_info=True
            )
            # Fallback to standard streaming
            async for chunk in self.standard_streaming_flow.run_stream(ctx):
                yield chunk
