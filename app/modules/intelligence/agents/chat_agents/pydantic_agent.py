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
from app.modules.intelligence.memory.compression_service import CompressionService
from app.modules.intelligence.provider.llm_config import get_context_window_for_model
from app.modules.intelligence.memory.token_counter import TokenCounterService

logger = setup_logger(__name__)


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
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
        
        # Context compression support
        self.compression_summary = ""  # Will hold summary after compression
        self.compression_service = CompressionService(
            llm_provider=llm_provider,
            tools=tools
        )
        # Compression threshold: 90% allows better utilization before compression
        # For 200K context window: triggers at 180K tokens
        self.compression_threshold_percentage = 0.90
        
        # Safety limit: Maximum compression cycles before breaking out
        self.max_compression_cycles = 5
        
        # Token counting support
        self.token_counter = TokenCounterService(llm_provider=llm_provider)

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
                    url=mcp_server["link"], timeout=10.0  # 10 second timeout
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

        # Prepare compression summary section if available
        # CRITICAL: Put this at the TOP, not buried after role/goal
        compression_prefix = ""
        compression_context = ""
        
        if self.compression_summary:  # If we have a compression summary from previous cycle
            # Extract key sections for top-level warning
            compression_prefix = """
            🛑🛑🛑 STOP - READ THIS FIRST - YOU ARE IN A CONTINUATION CYCLE 🛑🛑🛑
            
            DO NOT say "Let me first explore..." or "Let me examine..."
            DO NOT call fetch_file or get_file_content for files already read
            DO NOT ask questions that were already answered
            
            YOU MUST READ THE SUMMARY BELOW BEFORE DOING ANYTHING.
            
            ═══════════════════════════════════════════════════════════════
            """
            
            compression_context = f"""
            {self.compression_summary}
            
            ═══════════════════════════════════════════════════════════════
            ⚡ YOU ARE CONTINUING FROM WHERE YOU LEFT OFF ⚡
            ═══════════════════════════════════════════════════════════════
            
            """

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
            {compression_prefix}
            {compression_context}
            
            Role: {config.role}
            Goal: {config.goal}
            Backstory:
            {config.backstory}

            {multimodal_instructions}

            CURRENT CONTEXT AND AGENT TASK OVERVIEW:
            {self._create_task_description(task_config=config.tasks[0],ctx=ctx)}
            """,
            "result_type": str,
            "output_retries": 3,
            "output_type": str,
            "defer_model_check": True,
            "end_strategy": "exhaustive",
            "model_settings": {"max_tokens": 14000},
            "instrument": True,  # Enable Phoenix tracing for this agent
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
                except Exception as e:
                    logger.error(
                        f"Invalid base64 format for image {attachment_id}: {str(e)}"
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

            except Exception as e:
                logger.error(
                    f"Failed to add image {attachment_id} to content: {str(e)}",
                    exc_info=True,
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
                    except Exception as e:
                        logger.error(
                            f"Invalid base64 format for context image {attachment_id}: {str(e)}"
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
                except Exception as e:
                    logger.error(
                        f"Failed to add context image {attachment_id} to content: {str(e)}",
                        exc_info=True,
                    )
                    continue

        logger.info(
            f"Final multimodal content has {len(content)} elements: 1 text + {len(content)-1} images"
        )
        return content

    def _get_system_instructions_text(self) -> str:
        """
        Extract the system instructions text that will be sent to the agent.
        This extracts the EXACT instructions used in _create_agent() for token counting.
        
        IMPORTANT: This is READ-ONLY extraction for token counting purposes.
        It does NOT modify agent behavior or instructions in any way.
        
        Includes:
        - Compression summary (if present)
        - Role, goal, backstory
        
        Note: Task description and multimodal instructions are request-specific
        and counted separately via current message content.
        """
        try:
            config = self.config
            
            # Build compression context if available (matches lines 127-151 EXACTLY)
            compression_prefix = ""
            compression_context = ""
            
            if self.compression_summary:
                compression_prefix = """
            🛑🛑🛑 STOP - READ THIS FIRST - YOU ARE IN A CONTINUATION CYCLE 🛑🛑🛑
            
            DO NOT say "Let me first explore..." or "Let me examine..."
            DO NOT call fetch_file or get_file_content for files already read
            DO NOT ask questions that were already answered
            
            YOU MUST READ THE SUMMARY BELOW BEFORE DOING ANYTHING.
            
            ═══════════════════════════════════════════════════════════════
            """
                
                compression_context = f"""
            {self.compression_summary}
            
            ═══════════════════════════════════════════════════════════════
            ⚡ YOU ARE CONTINUING FROM WHERE YOU LEFT OFF ⚡
            ═══════════════════════════════════════════════════════════════
            
            """
            
            # Reconstruct instructions (matches lines 168-176 EXACTLY)
            instructions = f"""
        {compression_prefix}
        {compression_context}
        
        Role: {config.role}
        Goal: {config.goal}
        Backstory:
        {config.backstory}
        """
            
            return instructions.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract system instructions: {e}")
            return ""

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
            logger.error(f"Error in standard run method: {str(e)}", exc_info=True)
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

        except Exception as e:
            logger.error(f"Error in multimodal run method: {str(e)}", exc_info=True)
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

    def _check_token_threshold(
        self, 
        message_history: list,
        current_query: str,
        context_window: int, 
        token_threshold: int
    ) -> tuple[bool, int]:
        """
        Check if token usage exceeds threshold by using TokenCounterService.
        
        This method now delegates to the token counter service which handles:
        - Provider-specific token counting
        - Context window lookup
        - Threshold calculation
        
        Args:
            message_history: List of messages that have been sent (including tool results!)
            current_query: The current user query being processed
            context_window: Context window size (not used, kept for compatibility)
            token_threshold: Token threshold (not used, kept for compatibility)
        
        Returns:
            tuple: (should_compress: bool, total_tokens: int)
        """
        try:
            # Build complete message list including current query
            # Combine history with current query for accurate count
            all_messages = list(message_history)
            if current_query:
                # Add current query as a user message
                all_messages.append(ModelResponse([TextPart(content=current_query)]))
            
            if not all_messages:
                logger.debug("No messages to count")
                return (False, 0)
            
            # Extract system instructions for token counting (READ-ONLY extraction)
            system_instructions = self._get_system_instructions_text()
            
            # Get tools for token counting
            tools_list = self.tools  # List of StructuredTool objects
            
            # Use token counter service to check threshold
            # Pass context_window and token_threshold from caller to use their calculated values
            should_compress, token_count, actual_context_window = self.token_counter.check_token_threshold(
                messages=all_messages,
                system_instructions=system_instructions,
                tools=tools_list,
                model_name=None,  # Let service use configured model
                threshold_percentage=self.compression_threshold_percentage,
                config_type="chat",
                context_window=context_window,
                token_threshold=token_threshold
            )
            
            return (should_compress, token_count)
            
        except Exception as e:
            logger.error(f"Token threshold check failed: {e}", exc_info=True)
            # Don't trigger compression on error - continue execution
            return (False, 0)
    
    async def _perform_compression(
        self,
        ctx: ChatContext,
        run,
        message_history: list,
        compression_cycle: int,
        path_label: str = ""
    ) -> tuple[bool, list, int]:
        """
        Perform compression on the current run's message history.
        
        Args:
            ctx: Chat context
            run: Current agent run (not used, kept for API compatibility)
            message_history: Current message history to compress
            compression_cycle: Current compression cycle number
            path_label: Label for logging (e.g., "[Fallback]")
            
        Returns:
            tuple: (success: bool, new_message_history: list, new_cycle: int)
        """
        label = f"{path_label} " if path_label else ""
        logger.info(
            f"🗜️  {label}Starting compression process...\n"
            f"  Current cycle: {compression_cycle}\n"
            f"  Message history size: {len(message_history)}"
        )
        
        try:
            # Use the message_history directly (list of ModelResponse objects)
            # This is what the agent has been working with
            # Call compression service with retries
            summary = await self.compression_service.compress_message_history(
                messages=message_history,
                original_user_query=ctx.query,
                project_id=ctx.project_id,
                max_retries=2
            )
            
            # If compression failed, return failure
            if summary is None:
                logger.warning(f"⚠️  {label}Compression failed. Continuing with uncompressed history.")
                return (False, message_history, compression_cycle + 1)
            
            # Update compression summary for next iteration
            # This will be injected into agent instructions, not message history
            self.compression_summary = f"""**PREVIOUS EXECUTION SUMMARY**

You are continuing a task that was started earlier. Here's what you already accomplished:

{summary}

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL ANTI-LOOP INSTRUCTIONS ⚠️
═══════════════════════════════════════════════════════════════

**THIS SUMMARY IS ABSOLUTE TRUTH**: Everything in section 2 "BANNED ACTIONS" and section 3 "Technical Scratchpad" above is VERIFIED and COMPLETE. It is NOT a suggestion - it is DONE WORK.

**CRITICAL - READ SECTION 2A BANNED TOOL CALLS**:
- If a tool call is listed with 🚫 in section 2A, you are FORBIDDEN from calling it
- If a file is in section 2C "FILES ALREADY READ", do NOT fetch it again
- If a phrase is in section 2B "BANNED PHRASES", do NOT use it

**DO NOT REPEAT THESE ACTIONS**:
- ❌ DO NOT call any tool listed in section 2A with 🚫
- ❌ DO NOT use any phrase listed in section 2B with 🚫
- ❌ DO NOT say "Let's start by exploring..." for anything in section 2C
- ❌ DO NOT say "Let's examine..." for anything in section 2C
- ❌ DO NOT re-read files listed in section 2C "FILES ALREADY READ"
- ❌ DO NOT regenerate code listed in section 2C "CODE ALREADY WRITTEN"

**WHAT YOU SHOULD DO**:
✅ Read section 4 "What To Do NEXT" - that's your ONLY task
✅ Use the technical details from section 3 as FACTS - don't re-verify them
✅ If you need NEW information not in the summary, use tools
✅ Move FORWARD with implementation, not BACKWARD with re-exploration
✅ If code was already generated in section 2, your job is to EXECUTE/TEST it, not regenerate it

**PHASE TRANSITIONS** (Important for Progress):
- If section 2 shows exploration was done → Don't explore again, move to design
- If section 2 shows code was generated → Don't regenerate, move to testing/execution/next feature
- If section 2 shows testing was done → Don't test again, move to documentation/deployment/next step

**IF YOU CATCH YOURSELF**:
- About to say "Let me first explore..." → STOP! Check section 2B BANNED PHRASES
- About to call fetch_file or get_file_content → STOP! Check section 2A BANNED TOOL CALLS
- About to generate code → STOP! Check section 2C "CODE ALREADY WRITTEN"
- Starting from scratch → STOP! You are NOT starting fresh, you are CONTINUING"""
            
            # Keep only recent messages (no duplicate summary in history)
            # Reduced from 8 to 3 messages to save more tokens
            recent_history = [
                ModelResponse([TextPart(content=msg)]) 
                for msg in ctx.history[-3:]
            ]
            
            new_message_history = recent_history
            
            logger.info(
                f"✅ {label}Compression complete:\n"
                f"  Summary: {len(summary)} chars\n"
                f"  New history: {len(new_message_history)} messages\n"
                f"  Restarting agent cycle #{compression_cycle + 2}..."
            )
            
            return (True, new_message_history, compression_cycle + 1)
            
        except Exception as compression_error:
            logger.error(
                f"❌ {label}Compression process failed: {compression_error}",
                exc_info=True
            )
            return (False, message_history, compression_cycle + 1)

    async def _run_multimodal_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multimodal response using PydanticAI's native capabilities"""
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
                                    yield ChatAgentResponse(
                                        response=event.part.content,
                                        tool_calls=[],
                                        citations=[],
                                    )
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    yield ChatAgentResponse(
                                        response=event.delta.content_delta,
                                        tool_calls=[],
                                        citations=[],
                                    )

                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    yield ChatAgentResponse(
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=event.part.tool_call_id or "",
                                                event_type=ToolCallEventType.CALL,
                                                tool_name=event.part.tool_name,
                                                tool_response=get_tool_run_message(
                                                    event.part.tool_name
                                                ),
                                                tool_call_details={
                                                    "summary": get_tool_call_info_content(
                                                        event.part.tool_name,
                                                        event.part.args_as_dict(),
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

        except Exception as e:
            logger.error(f"Error in multimodal stream: {str(e)}", exc_info=True)
            # Fallback to standard streaming
            async for chunk in self._run_standard_stream(ctx):
                yield chunk

    async def _run_standard_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Standard streaming execution with looping context compression"""
        
        # Get model info for token tracking
        model_name = self.llm_provider.chat_config.model
        context_window = get_context_window_for_model(model_name)
        token_threshold = int(context_window * self.compression_threshold_percentage)
        
        logger.info(
            f"🚀 Starting agent with compression enabled:\n"
            f"  Model: {model_name}\n"
            f"  Context window: {context_window:,} tokens\n"
            f"  Compression threshold: {token_threshold:,} tokens"
        )
        
        # Initialize message history from context
        message_history = [
            ModelResponse([TextPart(content=msg)]) for msg in ctx.history
        ]
        
        compression_cycle = 0
        
        # Main compression loop
        while True:
            # Safety check: break out if we've hit max compression cycles
            if compression_cycle >= self.max_compression_cycles:
                logger.error(
                    f"🛑 Max compression cycles ({self.max_compression_cycles}) reached. "
                    f"Breaking out to prevent infinite loop."
                )
                yield ChatAgentResponse(
                    response=f"\n\n*[ERROR: Maximum compression cycles ({self.max_compression_cycles}) reached. "
                    f"The task may be too complex for the current context window. Please try breaking it into smaller subtasks.]*\n\n",
                    tool_calls=[],
                    citations=[]
                )
                return
            
            # Create agent with compression summary (empty first time, populated after compression)
            agent = self._create_agent(ctx)
            
            logger.info(
                f"🔄 Agent cycle #{compression_cycle + 1}/{self.max_compression_cycles}. "
                f"Compression summary length: {len(self.compression_summary)} chars"
            )
            
            # Warn if approaching limit
            if compression_cycle >= self.max_compression_cycles - 1:
                logger.warning(
                    f"⚠️  Approaching max compression cycles! "
                    f"This is cycle {compression_cycle + 1}/{self.max_compression_cycles}"
                )
            
            compression_needed = False
            
            try:
                # Try to initialize MCP servers with timeout handling
                try:
                    async with agent.run_mcp_servers():
                        async with agent.iter(
                            user_prompt=ctx.query,  # Always the original query
                            message_history=message_history,
                        ) as run:
                            async for node in run:
                                if Agent.is_model_request_node(node):
                                    # A model request node => We can stream tokens from the model's request
                                    model_response_text = []  # Track model's text output
                                    try:
                                        async with node.stream(run.ctx) as request_stream:
                                            async for event in request_stream:
                                                if isinstance(
                                                    event, PartStartEvent
                                                ) and isinstance(event.part, TextPart):
                                                    model_response_text.append(event.part.content)
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
                                                    model_response_text.append(event.delta.content_delta)
                                                    yield ChatAgentResponse(
                                                        response=event.delta.content_delta,
                                                        tool_calls=[],
                                                        citations=[],
                                                    )
                                        
                                        # Add model's text response to message_history for cumulative tracking
                                        if model_response_text:
                                            full_response = "".join(model_response_text)
                                            message_history.append(
                                                ModelResponse([TextPart(content=full_response)])
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
                                    except Exception as e:
                                        logger.error(
                                            f"Unexpected error in model request stream: {e}"
                                        )
                                        yield ChatAgentResponse(
                                            response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        continue

                                elif Agent.is_call_tools_node(node):
                                    tool_results_in_this_cycle = []
                                    try:
                                        async with node.stream(run.ctx) as handle_stream:
                                            async for event in handle_stream:
                                                if isinstance(event, FunctionToolCallEvent):
                                                    
                                                    yield ChatAgentResponse(
                                                        response="",
                                                        tool_calls=[
                                                            ToolCallResponse(
                                                                call_id=event.part.tool_call_id
                                                                or "",
                                                                event_type=ToolCallEventType.CALL,
                                                                tool_name=event.part.tool_name,
                                                                tool_response=get_tool_run_message(
                                                                    event.part.tool_name
                                                                ),
                                                                tool_call_details={
                                                                    "summary": get_tool_call_info_content(
                                                                        event.part.tool_name,
                                                                        event.part.args_as_dict(),
                                                                    )
                                                                },
                                                            )
                                                        ],
                                                        citations=[],
                                                    )
                                                if isinstance(
                                                    event, FunctionToolResultEvent
                                                ):
                                                    # Track tool result content for token counting
                                                    result_content = str(event.result.content) if event.result.content else ""
                                                    tool_results_in_this_cycle.append(result_content)
                                                    
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
                                        
                                        # Check tokens AFTER tool execution completes
                                        # Add tool results to message_history for cumulative tracking
                                        for result_content in tool_results_in_this_cycle:
                                            message_history.append(
                                                ModelResponse([TextPart(content=result_content)])
                                            )
                                        
                                        # Now check tokens on the accumulated history
                                        should_compress, current_tokens = self._check_token_threshold(
                                            message_history, ctx.query, context_window, token_threshold
                                        )
                                        
                                        if should_compress:
                                            compression_needed = True
                                            break  # Exit node loop to trigger compression
                                            
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
                                    except Exception as e:
                                        logger.error(
                                            f"Unexpected error in tool call stream: {e}"
                                        )
                                        yield ChatAgentResponse(
                                            response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        continue

                                elif Agent.is_end_node(node):
                                    logger.info("✅ Agent execution completed successfully")
                                    return  # Exit the while True loop - we're done!
                            
                            # If we exited the loop naturally (not compression), we're done
                            if not compression_needed:
                                logger.info("Agent completed without needing compression")
                                return
                    
                    # === PERFORM COMPRESSION (outside agent.iter context) ===
                    if compression_needed and run:
                        yield ChatAgentResponse(
                            response="\n\n*[Context window filling up - compressing history...]*\n\n",
                            tool_calls=[], citations=[]
                        )
                        
                        # Call compression helper
                        success, message_history, compression_cycle = await self._perform_compression(
                            ctx, run, message_history, compression_cycle
                        )
                        
                        if success:
                            yield ChatAgentResponse(
                                response="*[Compression complete. Continuing with fresh context...]*\n\n",
                                tool_calls=[], citations=[]
                            )
                        else:
                            yield ChatAgentResponse(
                                response="*[Compression unavailable. Continuing with full history...]*\n\n",
                                tool_calls=[], citations=[]
                            )
                        
                        continue  # Loop continues - will create new agent

                except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                    logger.warning(f"MCP server initialization failed: {mcp_error}")
                    logger.info("Continuing without MCP servers...")

                    # Fallback: run without MCP servers (with same compression logic)
                    try:
                        async with agent.iter(
                            user_prompt=ctx.query,
                            message_history=message_history,
                        ) as run:
                            async for node in run:
                                if Agent.is_model_request_node(node):
                                    model_response_text = []  # Track model's text output
                                    try:
                                        async with node.stream(run.ctx) as request_stream:
                                            async for event in request_stream:
                                                if isinstance(
                                                    event, PartStartEvent
                                                ) and isinstance(event.part, TextPart):
                                                    model_response_text.append(event.part.content)
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
                                                    model_response_text.append(event.delta.content_delta)
                                                    yield ChatAgentResponse(
                                                        response=event.delta.content_delta,
                                                        tool_calls=[],
                                                        citations=[],
                                                    )
                                        
                                        # Add model's text response to message_history for cumulative tracking
                                        if model_response_text:
                                            full_response = "".join(model_response_text)
                                            message_history.append(
                                                ModelResponse([TextPart(content=full_response)])
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
                                    except Exception as e:
                                        logger.error(
                                            f"Unexpected error in fallback model request stream: {e}"
                                        )
                                        yield ChatAgentResponse(
                                            response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        continue

                                elif Agent.is_call_tools_node(node):
                                    tool_results_in_this_cycle = []
                                    try:
                                        async with node.stream(run.ctx) as handle_stream:
                                            async for event in handle_stream:
                                                if isinstance(event, FunctionToolCallEvent):
                                                    
                                                    yield ChatAgentResponse(
                                                        response="",
                                                        tool_calls=[
                                                            ToolCallResponse(
                                                                call_id=event.part.tool_call_id
                                                                or "",
                                                                event_type=ToolCallEventType.CALL,
                                                                tool_name=event.part.tool_name,
                                                                tool_response=get_tool_run_message(
                                                                    event.part.tool_name
                                                                ),
                                                                tool_call_details={
                                                                    "summary": get_tool_call_info_content(
                                                                        event.part.tool_name,
                                                                        event.part.args_as_dict(),
                                                                    )
                                                                },
                                                            )
                                                        ],
                                                        citations=[],
                                                    )
                                                if isinstance(
                                                    event, FunctionToolResultEvent
                                                ):
                                                    # Track tool result content for token counting
                                                    result_content = str(event.result.content) if event.result.content else ""
                                                    tool_results_in_this_cycle.append(result_content)
                                                    
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
                                        
                                        # Check tokens AFTER tool execution completes
                                        # Add tool results to message_history for cumulative tracking
                                        for result_content in tool_results_in_this_cycle:
                                            message_history.append(
                                                ModelResponse([TextPart(content=result_content)])
                                            )
                                        
                                        # Now check tokens on the accumulated history
                                        should_compress, current_tokens = self._check_token_threshold(
                                            message_history, ctx.query, context_window, token_threshold
                                        )
                                        
                                        if should_compress:
                                            compression_needed = True
                                            break  # Exit node loop to trigger compression
                                            
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
                                    except Exception as e:
                                        logger.error(
                                            f"Unexpected error in fallback tool call stream: {e}"
                                        )
                                        yield ChatAgentResponse(
                                            response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        continue

                                elif Agent.is_end_node(node):
                                    logger.info("✅ Agent execution completed successfully")
                                    return
                            
                            # If we exited the loop naturally (not compression), we're done
                            if not compression_needed:
                                logger.info("Agent completed without needing compression (fallback path)")
                                return
                        
                        # === PERFORM COMPRESSION IN FALLBACK PATH ===
                        if compression_needed and run:
                            yield ChatAgentResponse(
                                response="\n\n*[Context window filling up - compressing history...]*\n\n",
                                tool_calls=[], citations=[]
                            )
                            
                            # Call compression helper with fallback label
                            success, message_history, compression_cycle = await self._perform_compression(
                                ctx, run, message_history, compression_cycle, path_label="[Fallback]"
                            )
                            
                            if success:
                                yield ChatAgentResponse(
                                    response="*[Compression complete. Continuing with fresh context...]*\n\n",
                                    tool_calls=[], citations=[]
                                )
                            else:
                                yield ChatAgentResponse(
                                    response="*[Compression unavailable. Continuing with full history...]*\n\n",
                                    tool_calls=[], citations=[]
                                )
                            
                            continue  # Loop continues - will create new agent

                    except (ModelRetry, AgentRunError, UserError) as pydantic_error:
                        logger.error(
                            f"Pydantic-ai error in fallback agent iteration: {pydantic_error}"
                        )
                        yield ChatAgentResponse(
                            response=f"\n\n*The agent encountered an error while processing your request: {str(pydantic_error)}*\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                        return  # Exit on error
                    except Exception as e:
                        logger.error(f"Unexpected error in fallback agent iteration: {e}")
                        yield ChatAgentResponse(
                            response=f"\n\n*An unexpected error occurred: {str(e)}*\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                        return  # Exit on error

            except (ModelRetry, AgentRunError, UserError) as pydantic_error:
                logger.error(
                    f"Pydantic-ai error in agent execution loop: {str(pydantic_error)}",
                    exc_info=True,
                )
                yield ChatAgentResponse(
                    response=f"\n\n*The agent encountered an error: {str(pydantic_error)}*\n\n",
                    tool_calls=[],
                    citations=[],
                )
                return  # Exit on error
            except Exception as e:
                logger.error(f"Error in agent execution loop: {str(e)}", exc_info=True)
                yield ChatAgentResponse(
                    response="\n\n*An error occurred during streaming*\n\n",
                    tool_calls=[],
                    citations=[],
                )
                return  # Exit on error
