import functools
import re
from typing import List, AsyncGenerator
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

    def _create_agent(self, ctx: ChatContext) -> Agent:
        config = self.config
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

    def _create_agent(self, ctx: ChatContext) -> Agent:
        config = self.config
        return Agent(
            model=self.llm_provider.get_pydantic_model(),
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=tool.func,  # type: ignore
                )
                for tool in self.tools
            ],
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
            """

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        logger.info("running pydantic-ai agent")
        try:
            # Create all tasks

            resp = await self._create_agent(ctx).run(
                user_prompt=ctx.query,
                message_history=[
                    ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                ],
            )

            # session and session.end_session("Success")

            return ChatAgentResponse(
                response=resp.output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info("running pydantic-ai agent stream")
        try:
            async with self._create_agent(ctx).iter(
                user_prompt=ctx.query,
                message_history=[
                    ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                ],
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
                        logger.info("result streamed successfully!!")

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e
