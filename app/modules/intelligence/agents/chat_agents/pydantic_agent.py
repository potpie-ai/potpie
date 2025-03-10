import re
from typing import List, AsyncGenerator

from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .crewai_agent import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger

from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from pydantic import BaseModel

from pydantic_ai import Agent, Tool
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
            model="openai:gpt-4o",
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
                                        citations=[],
                                    )

                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    yield ChatAgentResponse(
                                        # response=f"\n[Tools] The LLM calls tool={event.part.tool_name!r} {event.part.args_as_json_str}\n",
                                        response=f"\n<ToolCall>{event.part.tool_name}</ToolCall>\n",
                                        citations=[],
                                    )
                                if isinstance(event, FunctionToolResultEvent):
                                    yield ChatAgentResponse(
                                        # response=f"\n[Tools] Tool call {event.result.tool_name!r}\n",
                                        response=f"\n<ToolCallResult>{event.result.tool_name}</ToolCallResult>\n",
                                        citations=[],
                                    )

                    elif Agent.is_end_node(node):
                        logger.info("result streamed successfully!!")

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e
