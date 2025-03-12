from abc import abstractmethod
from contextlib import asynccontextmanager
import re
from typing import AsyncIterator, List, AsyncGenerator

from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentProvider,
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
from pydantic import BaseModel

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.usage import Usage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
)
from langchain_core.tools import StructuredTool

logger = setup_logger(__name__)


class ActionWrapper(BaseModel):
    action: str
    action_input: str


# class AgentResponse(StreamedResponse):
#     response: str


# class PydanticAIModel(Model):

#     def __init__(self):
#         pass

#     async def request(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> tuple[ModelResponse, Usage]:
#         """Make a request to the model."""
#         logger.info("Requesting to the model")
#         return ModelResponse(parts=[TextPart(content="")]), Usage()

#     @asynccontextmanager
#     async def request_stream(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> AsyncIterator[StreamedResponse]:
#         """Make a request to the model and return a streaming response."""

#         yield AgentResponse(response="streaming response")

#     @property
#     def model_name(self) -> str:
#         return "PydanticAI"

#     @property
#     def system(self) -> str | None:
#         """The system / model provider, ex: openai."""
#         return "PydanticAI"


def get_tool_run_message(tool_name: str):
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "Retrieving code from referenced names"
        case "Getcodechanges":
            return "Fetching code changes from your repo"
        case "GetNodesfromTags":
            return "Fetching nodes from tags"
        case "AskKnowledgeGraphQueries":
            return "Traversing the knowledge graph"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "Fetching code and docstrings"
        case "get_code_file_structure":
            return "Loading the file structure of the repo"
        case "GetNodeNeighboursFromNodeID":
            return "Identifying referenced code"
        case "WebpageContentExtractor":
            return "Querying information from the web"
        case "GitHubContentFetcher":
            return "Fetching content from github"
        case _:
            return "Fetching code"


def get_tool_response_message(tool_name: str):
    match tool_name:
        case "Getcodechanges":
            return "Code changes fetched successfully"
        case "GetNodesfromTags":
            return "Fetched nodes from tags"
        case "AskKnowledgeGraphQueries":
            return "Knowledge graph queries successful"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "Fetched code and docstrings"
        case "get_code_file_structure":
            return "File structure of the repo loaded successfully"
        case "GetNodeNeighboursFromNodeID":
            return "Fetched referenced code"
        case "WebpageContentExtractor":
            return "Information retrieved from web"
        case "GitHubContentFetcher":
            return "File contents fetched from github"
        case _:
            return "Code fetched"


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

        # tool name can't have spaces for langgraph agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        self.agent = Agent(
            # model=llm_provider.get_small_llm(AgentProvider.PYDANTICAI),  # type: ignore
            model="gpt-4o-mini",
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    function=tool.func,  # type: ignore
                )
                for tool in tools
            ],
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
            result_type=str,
            retries=3,
            result_retries=3,
            model_settings={
                "parallel_tool_calls": True,
            },
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
            node_ids = [ctx.node_ids]

        return f"""
                CONTEXT:
                User Query: {ctx.query}
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}

                Consider the chat history for any specific instructions or context: {" ,".join(ctx.history) if len(ctx.history) > 0 else "no chat history"}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                TASK:
                {task_config.description}

                Expected Output:
                {task_config.expected_output}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. Process and synthesize the gathered information
                3. Format your response in markdown, make sure it's well formatted
                4. Include relevant code snippets and file references
                5. Provide clear explanations
                6. Verify your output before submitting

                IMPORTANT:
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below

                With above information answer the user query: {ctx.query}
            """

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        try:
            # agentops.init(
            #     os.getenv("AGENTOPS_API_KEY"), default_tags=["openai-gpt-notebook"]
            # )
            # session = agentops.start_session()
            # Create all tasks

            task = self._create_task_description(self.tasks[0], ctx)

            resp = await self.agent.run(user_prompt=task)

            # session and session.end_session("Success")

            return ChatAgentResponse(
                response=resp.data,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        task = self._create_task_description(self.tasks[0], ctx)
        try:
            async with self.agent.iter(
                user_prompt=task,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        # A model request node => We can stream tokens from the model's request
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:

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
                                        # response=f"\n[Tools] The LLM calls tool={event.part.tool_name!r} {event.part.args_as_json_str}\n",
                                        response="",
                                        tool_calls=[
                                            ToolCallResponse(
                                                call_id=event.part.tool_call_id or "",
                                                event_type=ToolCallEventType.CALL,
                                                tool_name=event.part.tool_name,
                                                tool_response=get_tool_run_message(
                                                    event.part.tool_name
                                                ),
                                                tool_call_details={},
                                            )
                                        ],
                                        citations=[],
                                    )
                                if isinstance(event, FunctionToolResultEvent):
                                    yield ChatAgentResponse(
                                        # response=f"\n[Tools] Tool call {event.result.tool_name!r}\n",
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
                                                tool_call_details={},
                                            )
                                        ],
                                        citations=[],
                                    )

                    elif Agent.is_end_node(node):
                        logger.info("result streamed successfully!!")

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e
