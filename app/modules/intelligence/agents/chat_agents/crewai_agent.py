import json
from typing import Any, List, AsyncGenerator, Optional

from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentProvider,
)
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from ..chat_agent import ChatAgent, ChatAgentResponse, ChatContext

logger = setup_logger(__name__)


class TaskConfig(BaseModel):
    """Model for task configuration from agent_config.json"""

    description: str
    expected_output: str
    context: Optional[Task] = None


class AgentConfig(BaseModel):
    """Model for agent configuration from agent_config.json"""

    role: str
    goal: str
    backstory: str
    tasks: List[TaskConfig]
    max_iter: int = 15


class CrewAIResponse(BaseModel):
    response: str = Field(
        ...,
        description="Full response to the query",
    )
    citations: List[str] = Field(
        ...,
        description="List of file names extracted from context and referenced in the response",
    )


class CrewAIAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[Any],
    ):
        """Initialize the agent with configuration and tools"""

        self.tasks = config.tasks
        self.max_iter = config.max_iter

        self.tools = tools
        self.tools_description = self._create_tools_description(tools)

        self.agent = Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=llm_provider.get_llm(AgentProvider.CREWAI),
            max_iter=config.max_iter,
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
            ctx.node_ids = [ctx.node_ids]

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
                {self.tools_description}

                **Output Requirements:**
                - Ensure that your final response MUST be a valid JSON object which follows the structure outlined in the Pydantic model: {ChatAgentResponse.model_json_schema()}
                - Do not wrap the response in ```json, ```python, ```code, or ``` symbols.
                - For citations, include only the `file_path` of the nodes fetched and used.
                - Do not include any explanation or additional text outside of this JSON object.
                - Ensure all of the expected output and code are included within the "response" string.

                With above information answer the user query: {ctx.query}
            """

    async def _create_task(self, task_config: TaskConfig, ctx: ChatContext) -> Task:
        """Create a task with proper context and description"""

        task_description = self._create_task_description(task_config, ctx)

        # Create task with agent
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output=task_config.expected_output,
            output_pydantic=CrewAIResponse,
        )
        if task_config.context:
            task.context = [task_config.context]

        return task

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        try:
            logger.info("running crew-ai agent")
            # agentops.init(
            #     os.getenv("AGENTOPS_API_KEY"), default_tags=["openai-gpt-notebook"]
            # )
            # Create all tasks
            tasks = []
            for i, task_config in enumerate(self.tasks):
                if len(tasks) > 0:
                    task_config.context = tasks[-1]
                task = await self._create_task(task_config, ctx)
                tasks.append(task)
                logger.info(f"Created task {i+1}/{len(self.tasks)}")

            # Create single crew with all tasks
            crew = Crew(
                agents=[self.agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

            logger.info(f"Starting Crew AI kickoff with {len(tasks)} tasks")
            result = await crew.kickoff_async()
            response = result.tasks_output[-1].raw
            pydantic_response: CrewAIResponse = result.tasks_output[-1].pydantic
            return ChatAgentResponse(
                response=response,
                tool_calls=[],
                citations=pydantic_response.citations,
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        logger.info("running crew-ai agent stream")
        yield await self.run(ctx)  # CrewAI doesn't support streaming response
