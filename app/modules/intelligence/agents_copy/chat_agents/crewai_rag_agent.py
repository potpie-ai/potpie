import json
from typing import Any, List, Optional, AsyncGenerator
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentType,
)
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel
from app.modules.utils.logger import setup_logger
from ..chat_agent import ChatAgent, ChatAgentResponse

logger = setup_logger(__name__)


class TaskConfig(BaseModel):
    """Model for task configuration from agent_config.json"""

    description: str
    expected_output: str


class AgentConfig(BaseModel):
    """Model for agent configuration from agent_config.json"""

    role: str
    goal: str
    backstory: str
    tasks: List[TaskConfig]
    max_iter: int = 5


class CrewAIRagAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        project_id: str,
        config: AgentConfig,
        tools: List[Any],  # TODO: Add type
    ):
        """Initialize the agent with configuration and tools"""

        self.project_id = project_id
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
            llm=llm_provider.get_small_llm(AgentType.CREWAI),
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
        query: str,
        node_ids: Optional[List[str]] = None,
        context: str = "",
    ) -> str:
        """Create a task description from task configuration"""
        if isinstance(node_ids, str):
            node_ids = [node_ids]

        return f"""
                CONTEXT:
                Query: {query}
                Project ID: {self.project_id}
                Node IDs: {node_ids}
                Additional Context: {context}

                TASK:
                {task_config.description}

                Expected Output Format:
                {json.dumps(task_config.expected_output, indent=2)}

                INSTRUCTIONS:
                1. Use the available tools to gather information
                2. Process and synthesize the gathered information
                3. Format your response in markdown
                4. Include relevant code snippets and file references
                5. Provide clear explanations

                IMPORTANT:
                - You have a maximum of {self.max_iter} iterations
                - Use tools efficiently and avoid unnecessary API calls
                - Only use the tools listed below
                {self.tools_description}
            """

    async def _create_task(
        self,
        query: str,
        task_config: TaskConfig,
        node_ids: Optional[List[str]] = None,
        context: str = "",
    ) -> Task:
        """Create a task with proper context and description"""

        task_description = self._create_task_description(
            task_config, query, node_ids, context
        )

        # Create task with agent
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="Markdown formatted response with code context and explanations",
            output_pydantic=ChatAgentResponse,
        )

        return task

    async def run(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        return await anext(self.run_stream(query, history, node_ids))

    async def run_stream(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Main execution flow"""
        try:
            # Create all tasks
            tasks = []
            for i, task_config in enumerate(self.tasks):
                task = await self._create_task(query, task_config, node_ids)
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
            response: ChatAgentResponse = result.tasks_output[0].pydantic
            yield response

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e
