import functools
import re
from typing import List, AsyncGenerator

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
from .crewai_agent import AgentConfig, TaskConfig
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
)
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError
from langchain_core.tools import StructuredTool

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

    def _create_agent(self, ctx: ChatContext) -> Agent:
        config = self.config

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

        return Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=handle_exception(tool.func),  # type: ignore
                )
                for tool in self.tools
            ],
            mcp_servers=mcp_toolsets,
            instructions=f"""
            Role: {config.role}
            Goal: {config.goal}
            Backstory:
            {config.backstory}
            CURRENT CONTEXT AND AGENT TASK OVERVIEW:
            {self._create_task_description(task_config=config.tasks[0],ctx=ctx)}
            """,
            result_type=str,
            output_retries=3,
            output_type=str,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"max_tokens": 14000},
        )

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

        return f"""
                CONTEXT:
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}
                Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                TASK HANDLING: (follow the method below if the user asks you to execute your task for your role and goal)
                {task_config.description}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. Process and synthesize the gathered information
                3. Format your response in markdown unless explicitely asked to output in a different format, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. Provide clear explanations
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                - You have access to tools in MCP Servers too, use them effectively. These mcp servers provide you with tools user might ask you to perform tasks on
            """

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        logger.info("running pydantic-ai agent")

        try:
            agent = self._create_agent(ctx)

            # Try to initialize MCP servers with timeout handling
            try:
                async with agent.run_mcp_servers():
                    resp = await agent.run(
                        user_prompt=ctx.query,
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in ctx.history
                        ],
                    )
            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                resp = await agent.run(
                    user_prompt=ctx.query,
                    message_history=[
                        ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                    ],
                )

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                tool_calls=[],
                citations=[],
            )

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info("running pydantic-ai agent stream")

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
                                logger.info("result streamed successfully!!")

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
                                logger.info("result streamed successfully!!")

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
                    logger.error(f"Unexpected error in fallback agent iteration: {e}")
                    yield ChatAgentResponse(
                        response=f"\n\n*An unexpected error occurred: {str(e)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )

        except (ModelRetry, AgentRunError, UserError) as pydantic_error:
            logger.error(
                f"Pydantic-ai error in run_stream method: {str(pydantic_error)}",
                exc_info=True,
            )
            yield ChatAgentResponse(
                response=f"\n\n*The agent encountered an error: {str(pydantic_error)}*\n\n",
                tool_calls=[],
                citations=[],
            )
        except Exception as e:
            logger.error(f"Error in run_stream method: {str(e)}", exc_info=True)
            yield ChatAgentResponse(
                response="\n\n*An error occurred during streaming*\n\n",
                tool_calls=[],
                citations=[],
            )
