import functools
import inspect
import re
from typing import List, AsyncGenerator, Sequence

import anyio

from pydantic_ai.mcp import MCPServerStreamableHTTP

from .tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .agent_config import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger
from app.modules.intelligence.tools.reasoning_manager import (
    _get_reasoning_manager,
    _reset_reasoning_manager,
)

from ..chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ModelResponse,
    TextPart,
    ImageUrl,
    UserContent,
    ModelMessage,
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError
from langchain_core.tools import StructuredTool

logger = setup_logger(__name__)


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception:
            # Use Loguru's native exception() with context kwargs
            logger.exception("Exception in tool function", tool_name=tool_func.__name__)
            return "An internal error occurred. Please try again later."

    return wrapper


class PydanticRagAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None = None,
    ):
        """Initialize the agent with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter

        # tool name can't have spaces for langgraph/pydantic agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        self.llm_provider = llm_provider
        self.tools = tools
        self.config = config
        self.mcp_servers = mcp_servers or []
        # Initialize history processor for token-aware context management
        from .history_processor import create_history_processor

        self._history_processor = create_history_processor(llm_provider)

    def _create_agent(self, ctx: ChatContext) -> Agent:
        config = self.config

        # Prepare multimodal instructions if images are present
        multimodal_instructions = self._prepare_multimodal_instructions(ctx)
        # Create MCP servers directly - continue even if some fail
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for mcp_server in self.mcp_servers:
            try:
                # Add timeout and connection handling for MCP servers
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_server["link"],
                    timeout=10.0,  # 10 second timeout
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

        logger.info(
            f"Created {len(mcp_toolsets)} MCP servers out of {len(self.mcp_servers)} configured"
        )

        allow_parallel_tools = self.llm_provider.chat_config.capabilities.get(
            "supports_tool_parallelism", True
        )

        agent_kwargs = {
            "model": self.llm_provider.get_pydantic_model(),
            "tools": [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=handle_exception(tool.func),  # type: ignore
                )
                for tool in self.tools
            ],
            "mcp_servers": mcp_toolsets,
            "instructions": f"""
# Agent Execution Guidelines

You are an AI assistant that helps users with code analysis and tasks. Follow these principles:

1. **Be thorough**: Analyze code carefully before making recommendations
2. **Use tools effectively**: Leverage available tools to gather information
3. **Provide clear explanations**: Explain your reasoning and findings
4. **Handle errors gracefully**: If a tool fails, try alternative approaches

## Tool Usage Best Practices

- Use `fetch_file` with `with_line_numbers=true` for precise code references
- Use `ask_knowledge_graph_queries` for semantic code search
- Use `get_code_file_structure` to understand project layout
- Verify your findings before presenting conclusions

## Output Guidelines

- Structure responses with clear headings
- Include relevant code snippets with file paths
- Summarize key findings at the end

<!-- CACHE_BREAKPOINT -->

Your Identity:
Role: {config.role}
Goal: {config.goal}
Backstory:
{config.backstory}

{multimodal_instructions}

CURRENT CONTEXT AND AGENT TASK OVERVIEW:
{self._create_task_description(task_config=config.tasks[0],ctx=ctx)}
            """,
            "output_retries": 3,
            "output_type": str,
            "defer_model_check": True,
            "end_strategy": "exhaustive",
            "model_settings": {"max_tokens": 14000},
            "instrument": True,
            "history_processors": [self._history_processor],
        }

        if not allow_parallel_tools:
            try:
                signature = inspect.signature(Agent.__init__)
                if "allow_parallel_tool_calls" in signature.parameters:
                    agent_kwargs["allow_parallel_tool_calls"] = False
                elif "max_parallel_tool_calls" in signature.parameters:
                    agent_kwargs["max_parallel_tool_calls"] = 1
                elif "tool_parallelism" in signature.parameters:
                    agent_kwargs["tool_parallelism"] = False
                else:
                    logger.info(
                        "Parallel tool call disabling not supported by current pydantic-ai Agent signature."
                    )
            except Exception as signature_error:
                logger.warning(
                    "Failed to inspect Agent signature for parallel tool call support: %s",
                    signature_error,
                )

        return Agent(**agent_kwargs)

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
    ) -> str:
        """Create a task description from task configuration"""
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
                1. Use the available tools to gather information
                2. {"Analyze the provided images in detail and " if ctx.has_images() else ""}Process and synthesize the gathered information
                3. Format your response in markdown unless explicitely asked to output in a different format, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. {"Reference specific details from the images when relevant" if ctx.has_images() else "Provide clear explanations"}
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                - You have access to tools in MCP Servers too, use them effectively. These mcp servers provide you with tools user might ask you to perform tasks on
                {"- Provide detailed image analysis when images are present" if ctx.has_images() else ""}
            """

    def _debug_multimodal_content(self, ctx: ChatContext) -> None:
        """Debug method to log detailed information about multimodal content"""
        logger.info("=== MULTIMODAL CONTENT DEBUG ===")
        logger.info(f"Context has images: {ctx.has_images()}")

        if ctx.has_images():
            all_images = ctx.get_all_images()
            current_images = ctx.get_current_images_only()
            context_images = ctx.get_context_images_only()

            logger.info(f"Total images: {len(all_images)}")
            logger.info(f"Current images: {len(current_images)}")
            logger.info(f"Context images: {len(context_images)}")

            for img_id, img_data in all_images.items():
                logger.info(f"Image {img_id}:")
                logger.info(f"  - Type: {img_data.get('context_type', 'unknown')}")
                logger.info(f"  - File name: {img_data.get('file_name', 'unknown')}")
                logger.info(f"  - File size: {img_data.get('file_size', 'unknown')}")
                logger.info(f"  - MIME type: {img_data.get('mime_type', 'unknown')}")
                logger.info(f"  - Has base64: {'base64' in img_data}")
                if "base64" in img_data and isinstance(img_data["base64"], str):
                    base64_len = len(img_data["base64"])
                    logger.info(f"  - Base64 length: {base64_len}")

        # Test vision model detection
        is_vision = self.llm_provider.is_vision_model()
        logger.info(f"Current model supports vision: {is_vision}")

        logger.info("=== END MULTIMODAL DEBUG ===")

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

                    # Test if base64 is valid
                    base64.b64decode(base64_data)
                except Exception:
                    logger.exception(
                        f"Invalid base64 format for image {attachment_id}",
                        attachment_id=attachment_id,
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

            except Exception:
                logger.exception(
                    f"Failed to add image {attachment_id} to content",
                    attachment_id=attachment_id,
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
                    except Exception:
                        logger.exception(
                            f"Invalid base64 format for context image {attachment_id}",
                            attachment_id=attachment_id,
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
                except Exception:
                    logger.exception(
                        f"Failed to add context image {attachment_id} to content",
                        attachment_id=attachment_id,
                    )
                    continue

        logger.info(
            f"Final multimodal content has {len(content)} elements: 1 text + {len(content)-1} images"
        )
        return content

    async def _prepare_multimodal_message_history(
        self, ctx: ChatContext
    ) -> List[ModelMessage]:
        """Prepare message history with multimodal support"""
        history_messages = []

        for msg in ctx.history:
            # For now, keep history as text-only to avoid token bloat
            # Images are only added to the current query
            history_messages.append(ModelResponse([TextPart(content=str(msg))]))

        return history_messages

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow with multimodal support using PydanticAI's native capabilities"""
        logger.info(
            f"Running pydantic-ai agent {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Initialize code changes manager with conversation_id for persistence across messages
        from app.modules.intelligence.tools.code_changes_manager import (
            _init_code_changes_manager,
        )

        _init_code_changes_manager(ctx.conversation_id)
        logger.info(
            f"🔄 Initialized code changes manager for conversation_id={ctx.conversation_id}"
        )

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal"
            )
            return await self._run_multimodal(ctx)
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only"
                )
            # Use standard PydanticAI agent for text-only
            return await self._run_standard(ctx)

    async def _run_standard(self, ctx: ChatContext) -> ChatAgentResponse:
        """Standard text-only agent execution"""
        try:
            # Prepare message history
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run agent
            agent = self._create_agent(ctx)

            # Try to initialize MCP servers with timeout handling
            try:
                async with agent.run_mcp_servers():
                    resp = await agent.run(
                        user_prompt=ctx.query,
                        message_history=message_history,
                    )
            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                resp = await agent.run(
                    user_prompt=ctx.query,
                    message_history=message_history,
                )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.exception("Error in standard run method")
            return ChatAgentResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                tool_calls=[],
                citations=[],
            )

    async def _run_multimodal(self, ctx: ChatContext) -> ChatAgentResponse:
        """Multimodal agent execution using PydanticAI's native multimodal capabilities"""
        try:
            # Debug multimodal content
            self._debug_multimodal_content(ctx)

            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create and run agent
            agent = self._create_agent(ctx)

            resp = await agent.run(
                user_prompt=multimodal_content,
                message_history=message_history,
            )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception:
            logger.exception("Error in multimodal run method")
            # Fallback to standard execution
            logger.info("Falling back to standard text-only execution")
            return await self._run_standard(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info(
            f"Running pydantic-ai agent stream {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal streaming"
            )
            async for chunk in self._run_multimodal_stream(ctx):
                yield chunk
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only streaming"
                )
            # Use standard PydanticAI streaming for text-only
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_multimodal_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multimodal response using PydanticAI's native capabilities"""
        # Reset reasoning manager for this run
        _reset_reasoning_manager()
        reasoning_manager = _get_reasoning_manager()

        try:
            # Debug multimodal content
            self._debug_multimodal_content(ctx)

            # Create multimodal user content with images
            multimodal_content = self._create_multimodal_user_content(ctx)

            # Prepare message history (text-only for now to avoid token bloat)
            message_history = await self._prepare_multimodal_message_history(ctx)

            # Create agent
            agent = self._create_agent(ctx)

            # Stream the response
            async with agent.iter(
                user_prompt=multimodal_content,
                message_history=message_history,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        # A model request node => We can stream tokens from the model's request
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

                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    tool_args = event.part.args_as_dict()
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=event.part.tool_call_id or "",
                                                event_type=ToolCallEventType.CALL,
                                                tool_name=event.part.tool_name,
                                                tool_response=get_tool_run_message(
                                                    event.part.tool_name, tool_args
                                                ),
                                                tool_call_details={
                                                    "summary": get_tool_call_info_content(
                                                        event.part.tool_name,
                                                        tool_args,
                                                    )
                                                },
                                            )
                                        ],
                                        citations=[],
                                    )
                                if isinstance(event, FunctionToolResultEvent):
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=event.result.tool_call_id or "",
                                                event_type=ToolCallEventType.RESULT,
                                                tool_name=event.result.tool_name
                                                or "unknown tool",
                                                tool_response=get_tool_response_message(
                                                    event.result.tool_name
                                                    or "unknown tool"
                                                ),
                                                tool_call_details={
                                                    "summary": get_tool_result_info_content(
                                                        event.result.tool_name
                                                        or "unknown tool",
                                                        event.result.content,
                                                    )
                                                },
                                            )
                                        ],
                                        citations=[],
                                    )

                    elif Agent.is_end_node(node):
                        logger.info("multimodal result streamed successfully!!")
                        # Finalize and save reasoning content
                        reasoning_hash = reasoning_manager.finalize_and_save()
                        if reasoning_hash:
                            logger.info(
                                f"Reasoning content saved with hash: {reasoning_hash}"
                            )

        except Exception:
            logger.exception("Error in multimodal stream")
            # Fallback to standard streaming
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_standard_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Standard streaming execution with MCP server support"""
        # Reset reasoning manager for this run
        _reset_reasoning_manager()
        reasoning_manager = _get_reasoning_manager()

        # Create agent directly
        agent = self._create_agent(ctx)

        try:
            # Try to initialize MCP servers with timeout handling
            try:
                async with agent.run_mcp_servers():
                    async with agent.iter(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # A model request node => We can stream tokens from the model's request
                                try:
                                    async with node.stream(run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, TextPart):
                                                # Accumulate TextPart content for reasoning dump
                                                reasoning_manager.append_content(
                                                    event.part.content
                                                )
                                                yield ChatAgentResponse(
                                                    response=event.part.content,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
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
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in model request stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while processing your request. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Model request stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in model request stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_call_tools_node(node):
                                try:
                                    async with node.stream(run.ctx) as handle_stream:
                                        async for event in handle_stream:
                                            if isinstance(event, FunctionToolCallEvent):
                                                tool_args = event.part.args_as_dict()
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.part.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.CALL,
                                                            tool_name=event.part.tool_name,
                                                            tool_response=get_tool_run_message(
                                                                event.part.tool_name,
                                                                tool_args,
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_call_info_content(
                                                                    event.part.tool_name,
                                                                    tool_args,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, FunctionToolResultEvent
                                            ):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.result.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.RESULT,
                                                            tool_name=event.result.tool_name
                                                            or "unknown tool",
                                                            tool_response=get_tool_response_message(
                                                                event.result.tool_name
                                                                or "unknown tool"
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_result_info_content(
                                                                    event.result.tool_name
                                                                    or "unknown tool",
                                                                    event.result.content,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in tool call stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while calling tools. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Tool call stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in tool call stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_end_node(node):
                                logger.info("result streamed successfully!!")
                                # Finalize and save reasoning content
                                reasoning_hash = reasoning_manager.finalize_and_save()
                                if reasoning_hash:
                                    logger.info(
                                        f"Reasoning content saved with hash: {reasoning_hash}"
                                    )

            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                try:
                    async with agent.iter(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                try:
                                    async with node.stream(run.ctx) as request_stream:
                                        async for event in request_stream:
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, TextPart):
                                                # Accumulate TextPart content for reasoning dump
                                                reasoning_manager.append_content(
                                                    event.part.content
                                                )
                                                yield ChatAgentResponse(
                                                    response=event.part.content,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
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
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in fallback model request stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while processing your request. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Model request stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in fallback model request stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_call_tools_node(node):
                                try:
                                    async with node.stream(run.ctx) as handle_stream:
                                        async for event in handle_stream:
                                            if isinstance(event, FunctionToolCallEvent):
                                                tool_args = event.part.args_as_dict()
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.part.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.CALL,
                                                            tool_name=event.part.tool_name,
                                                            tool_response=get_tool_run_message(
                                                                event.part.tool_name,
                                                                tool_args,
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_call_info_content(
                                                                    event.part.tool_name,
                                                                    tool_args,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, FunctionToolResultEvent
                                            ):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.result.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.RESULT,
                                                            tool_name=event.result.tool_name
                                                            or "unknown tool",
                                                            tool_response=get_tool_response_message(
                                                                event.result.tool_name
                                                                or "unknown tool"
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_result_info_content(
                                                                    event.result.tool_name
                                                                    or "unknown tool",
                                                                    event.result.content,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in fallback tool call stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while calling tools. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Tool call stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in fallback tool call stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_end_node(node):
                                logger.info("result streamed successfully!!")
                                # Finalize and save reasoning content
                                reasoning_hash = reasoning_manager.finalize_and_save()
                                if reasoning_hash:
                                    logger.info(
                                        f"Reasoning content saved with hash: {reasoning_hash}"
                                    )

                except (ModelRetry, AgentRunError, UserError) as pydantic_error:
                    logger.error(
                        f"Pydantic-ai error in fallback agent iteration: {pydantic_error}"
                    )
                    yield ChatAgentResponse(
                        response=f"\n\n*The agent encountered an error while processing your request: {str(pydantic_error)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                except Exception as e:
                    logger.exception("Unexpected error in fallback agent iteration")
                    yield ChatAgentResponse(
                        response=f"\n\n*An unexpected error occurred: {str(e)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )

        except (ModelRetry, AgentRunError, UserError) as pydantic_error:
            logger.exception("Pydantic-ai error in run_stream method")
            yield ChatAgentResponse(
                response=f"\n\n*The agent encountered an error: {str(pydantic_error)}*\n\n",
                tool_calls=[],
                citations=[],
            )
        except Exception:
            logger.exception("Error in run_stream method")
            yield ChatAgentResponse(
                response="\n\n*An error occurred during streaming*\n\n",
                tool_calls=[],
                citations=[],
            )
