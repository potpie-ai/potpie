import json
import os
import re
from typing import Any, List, AsyncGenerator, Union

import agentops
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentType,
)
from .crewai_agent import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from pydantic import BaseModel
import json

from langgraph.prebuilt import create_react_agent, ToolExecutor
from langchain_core.runnables import RunnableConfig


logger = setup_logger(__name__)


class ActionWrapper(BaseModel):
    action: str
    action_input: ChatAgentResponse


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

        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        self.tools = tools
        self.tools_description = self._create_tools_description(tools)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    config.backstory
                    + "\n you have following tools: {tool_names} with definitions: {tools}",
                ),
                ("human", "{input}\n\n{agent_scratchpad}"),
            ]
        )

        self.agent_1 = create_react_agent(
            model=llm_provider.get_large_llm(AgentType.LANGCHAIN),
            tools=ToolExecutor(tools),
            debug=True,
        )

    def _create_tools_description(self, tools: List[Any]):
        tools_desc = "Available Tools:\n"
        for tool in tools:
            tools_desc += f"{tool.name}: {tool.description}\n"
            if hasattr(tool, "args_schema"):
                schema = tool.args_schema.schema()
                tools_desc += (
                    f"Input: {json.dumps(schema.get('example', {}), indent=2)}\n\n"
                )
        return tools_desc

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
                - Ensure that your final response MUST be a valid JSON object which follows the structure outlined in the Pydantic model: {ActionWrapper.model_json_schema()}
                - Do not wrap the response in ```json, ```python, ```code, or ``` symbols.
                - For citations, include only the `file_path` of the nodes fetched and used.
                - Do not include any explanation or additional text outside of this JSON object.
                - Ensure all of the expected output and code are included within the "response" string.
                
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
            logger.info(f"given task is: {task}")
            # res = self.agent.run(
            #     {
            #         "input": task,
            #         '\n                "project_id"': ctx.project_id,
            #         '\n                "repo_name"': "potpie-ai/potpie",
            #         '\n                "url"': "https://www.github.com/potpie-ai/potpie",
            #     }
            # )

            res = self.agent_1.invoke(
                {"messages": task},
                config=RunnableConfig(recursion_limit=10),
            )

            logger.info("res====", res["messages"][-1])
            j_con = json.loads(res["messages"][-1].content)
            j_con["action_input"]

            # agent_executor = AgentExecutor(
            #     agent=self.agent,
            #     tools=self.tools,
            #     max_iterations=5,
            #     handle_parsing_errors=False,
            #     return_intermediate_steps=True,
            #     verbose=True,
            # )
            # res = await agent_executor.ainvoke({"input": task})

            # response: ChatAgentResponse = res["messages"][-1]["content"].action_input
            response = ChatAgentResponse(
                response=j_con["action_input"]["response"],
                citations=j_con["action_input"]["citations"],
            )
            # session and session.end_session("Success")

            return response

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        yield await self.run(ctx)  # CrewAI doesn't support streaming response
