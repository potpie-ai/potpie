import json
import re
from typing import Any, List, AsyncGenerator

from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentProvider,
)
from .crewai_agent import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger
from langchain_core.messages import AIMessage
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from pydantic import BaseModel

from langgraph.prebuilt import create_react_agent, ToolExecutor
from langchain_core.runnables import RunnableConfig


logger = setup_logger(__name__)


class ActionWrapper(BaseModel):
    action: str
    action_input: str


class LangchainRagAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[Any],
    ):
        """Initialize the agent with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter

        # tool name can't have spaces for langgraph agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        self.agent = create_react_agent(
            model=llm_provider.get_llm(AgentProvider.LANGCHAIN),  # type: ignore
            tools=ToolExecutor(tools),
            debug=True,
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

                Consider the chat history for any specific instructions or context: {" ,".join(ctx.history)}

                Additional Context:
                {ctx.additional_context}

                TASK:
                {task_config.description}

                Expected Output:
                {json.dumps(task_config.expected_output, indent=2)}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. Process and synthesize the gathered information
                3. Format your response in markdown
                4. Include relevant code snippets and file references
                5. Provide clear explanations

                IMPORTANT:
                - Respect the max iterations limit of {self.max_iter} when planning and executing tools.
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                - Respond with whatever info you have before you run out of max_iter

                **Output Requirements:**

                - All citations should be mentioned in comma seperated format at the end after "###Citations###" block
                    ex: If we have 3 citations, this should be at the end of the response:
                        `
                            ###Citations###
                            app/module/path1/file1.ext, app/module/path2/file2.ext, app/module/path3/file3.txt
                        `

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

            res = self.agent.invoke(
                {"messages": task},
                config=RunnableConfig(recursion_limit=self.max_iter),
            )

            # session and session.end_session("Success")

            return ChatAgentResponse(
                response=res["messages"][-1].content, citations=[], tool_calls=[]
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        task = self._create_task_description(self.tasks[0], ctx)
        try:
            for chunk in self.agent.stream(
                {"messages": task},
                config=RunnableConfig(recursion_limit=self.max_iter),
                stream_mode="updates",
            ):
                if (
                    chunk.get("agent")
                    and chunk["agent"].get("messages")
                    and len(chunk["agent"]["messages"]) > 0
                    and isinstance(chunk["agent"]["messages"][0], AIMessage)
                ):
                    yield ChatAgentResponse(
                        response=str(chunk["agent"]["messages"][0].content),
                        citations=[],
                        tool_calls=[],
                    )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e
