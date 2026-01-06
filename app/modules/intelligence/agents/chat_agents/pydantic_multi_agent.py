"""Refactored PydanticMultiAgent using modular components"""

import re
from typing import List, AsyncGenerator, Dict, Optional, Any
from langchain_core.tools import StructuredTool

from pydantic_ai import Agent

from app.modules.intelligence.provider.provider_service import ProviderService
from .agent_config import AgentConfig
from .history_processor import create_history_processor
from app.modules.utils.logger import setup_logger

from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext

# Import new modular components
from .multi_agent.utils.delegation_utils import AgentType
from .multi_agent.agent_factory import AgentFactory, create_default_delegate_agents
from .multi_agent.delegation_manager import DelegationManager
from .multi_agent.delegation_streamer import DelegationStreamer
from .multi_agent.stream_processor import StreamProcessor
from .multi_agent.execution_flows import (
    StandardExecutionFlow,
    MultimodalExecutionFlow,
    StreamingExecutionFlow,
    MultimodalStreamingExecutionFlow,
    reset_managers,
)
from .multi_agent.utils.tool_utils import create_error_response

# Export AgentType for backward compatibility
__all__ = ["PydanticMultiAgent", "AgentType"]

logger = setup_logger(__name__)


class PydanticMultiAgent(ChatAgent):
    """
    Multi-agent system using Pydantic AI with agent delegation patterns.

    This system consists of:
    1. A supervisor agent that coordinates tasks
    2. Specialized delegate agents for specific tasks
    3. Context passing between agents
    4. Usage tracking across all agents
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None = None,
        delegate_agents: Optional[Dict[AgentType, AgentConfig]] = None,
    ):
        """Initialize the multi-agent system with configuration and tools"""
        self.tasks = config.tasks
        self.max_iter = config.max_iter
        self.llm_provider = llm_provider
        self.tools = tools
        self.config = config
        self.mcp_servers = mcp_servers or []

        # Clean tool names (no spaces for pydantic agents)
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Initialize delegate agents
        self.delegate_agents = delegate_agents or create_default_delegate_agents()

        # Initialize history processor
        self._history_processor = create_history_processor(llm_provider)

        # Track current context and supervisor run for delegation
        self._current_context: Optional[ChatContext] = None
        self._current_supervisor_run_ref: Dict[str, Optional[Any]] = {"run": None}

        # Initialize delegation manager first (needs create_delegate_agent)
        # We'll create it with a placeholder, then update after agent_factory is ready
        delegation_manager = DelegationManager(
            create_delegate_agent=lambda agent_type, ctx: None,  # Placeholder
            delegation_streamer=None,  # Will be set after stream_processor is created
            create_error_response=create_error_response,
        )

        # Initialize agent factory (needs delegation_manager's create_delegation_function)
        # The create_delegation_function is created by delegation_manager, so we can use it
        self._agent_factory = AgentFactory(
            llm_provider=llm_provider,
            tools=tools,
            mcp_servers=mcp_servers,
            delegate_agents=self.delegate_agents,
            history_processor=self._history_processor,
            create_delegation_function=delegation_manager.create_delegation_function,
        )

        # Now update delegation_manager with real create_delegate_agent
        delegation_manager.create_delegate_agent = self._create_delegate_agent

        # Initialize stream processor with delegation_manager
        stream_processor = StreamProcessor(
            delegation_manager=delegation_manager,
            create_error_response=create_error_response,
        )

        # Initialize delegation streamer with stream processor's error handler
        delegation_streamer = DelegationStreamer(
            handle_stream_error=stream_processor.handle_stream_error,
            create_error_response=create_error_response,
            process_tool_call_node=stream_processor.process_tool_call_node,
        )

        # Update delegation_manager with delegation_streamer
        delegation_manager.delegation_streamer = delegation_streamer

        # Initialize execution flows
        standard_flow = StandardExecutionFlow(
            create_supervisor_agent=self._create_supervisor_agent,
            history_processor=self._history_processor,
        )

        multimodal_flow = MultimodalExecutionFlow(
            create_supervisor_agent=self._create_supervisor_agent,
            history_processor=self._history_processor,
            standard_flow=standard_flow,
        )

        streaming_flow = StreamingExecutionFlow(
            create_supervisor_agent=self._create_supervisor_agent,
            history_processor=self._history_processor,
            stream_processor=stream_processor,
            current_supervisor_run_ref=self._current_supervisor_run_ref,
        )

        multimodal_streaming_flow = MultimodalStreamingExecutionFlow(
            create_supervisor_agent=self._create_supervisor_agent,
            history_processor=self._history_processor,
            stream_processor=stream_processor,
            current_supervisor_run_ref=self._current_supervisor_run_ref,
            standard_streaming_flow=streaming_flow,
        )

        # Store flows
        self._standard_flow = standard_flow
        self._multimodal_flow = multimodal_flow
        self._streaming_flow = streaming_flow
        self._multimodal_streaming_flow = multimodal_streaming_flow

        # Store components for access
        self._delegation_manager = delegation_manager
        self._stream_processor = stream_processor

        # Reset managers on initialization
        reset_managers()

    def _create_supervisor_agent(self, ctx: ChatContext) -> Agent:
        """Create supervisor agent using agent factory"""
        return self._agent_factory.create_supervisor_agent(ctx, self.config)

    def _create_delegate_agent(self, agent_type: AgentType, ctx: ChatContext) -> Agent:
        """Create delegate agent using agent factory"""
        return self._agent_factory.create_delegate_agent(agent_type, ctx)

    def _handle_stream_error(
        self, error: Exception, context: str = "model request stream"
    ) -> Optional[ChatAgentResponse]:
        """Handle streaming errors - delegates to stream processor"""
        return self._stream_processor.handle_stream_error(error, context)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow with multi-agent coordination"""
        logger.info(
            f"Running pydantic multi-agent system {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Reset managers
        reset_managers()

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent"
            )
            return await self._multimodal_flow.run(ctx)
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only"
                )
            # Use standard PydanticAI multi-agent for text-only
            return await self._standard_flow.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream multi-agent response with delegation support"""
        logger.info(
            f"Running pydantic multi-agent stream {'with multimodal support' if ctx.has_images() else ''}"
        )

        # Store context for delegation functions
        self._current_context = ctx

        # Reset managers
        reset_managers()

        # Check if we have images and if the model supports vision
        if ctx.has_images() and self.llm_provider.is_vision_model():
            logger.info(
                f"Processing {len(ctx.get_all_images())} images with PydanticAI multimodal multi-agent streaming"
            )
            async for chunk in self._multimodal_streaming_flow.run_stream(ctx):
                yield chunk
        else:
            if ctx.has_images() and not self.llm_provider.is_vision_model():
                logger.warning(
                    "Images provided but current model doesn't support vision, proceeding with text-only streaming"
                )
            # Use standard PydanticAI multi-agent streaming for text-only
            async for chunk in self._streaming_flow.run_stream(ctx):
                yield chunk
