import json
import os
from typing import Any, Dict, List, Optional, Union

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.key_management.secret_manager import SecretManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class TaskExpectedOutput(BaseModel):
    """Model for expected output structure in task configuration"""

    output: str


class TaskConfig(BaseModel):
    """Model for task configuration from agent_config.json"""

    tools: List[str]
    description: str
    expected_output: Union[Dict[str, Any], str]

    @validator("expected_output", pre=True, always=True)
    def parse_expected_output(cls, v):
        """Ensure expected_output is a dictionary"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing expected_output: {str(e)}")
                raise ValueError("Invalid JSON format for expected_output")
        return v


class AgentConfig(BaseModel):
    """Model for agent configuration from agent_config.json"""

    user_id: str
    role: str
    goal: str
    backstory: str
    system_prompt: str
    tasks: List[TaskConfig]
    project_id: str = ""


class RuntimeAgent:
    def __init__(
        self,
        llm: Any,
        db: Session,
        agent_config: Dict[str, Any],
        secret_manager: SecretManager,
    ):
        """Initialize the agent with configuration and tools"""
        self.config = AgentConfig(**agent_config)
        self.secret_manager = secret_manager
        self.db = db
        self.llm = llm
        self.max_iter = int(os.getenv("MAX_ITER", "5"))

        self.user_id = self.config.user_id
        self.role = self.config.role
        self.goal = self.config.goal
        self.backstory = self.config.backstory

        self.project_id = None
        self.agent = None

        # Initialize tools
        self.tool_service = ToolService(db, self.user_id)
        self.tools = {}
        for tool_id, tool in self.get_available_tools().items():
            tool = self.tool_service.tools[tool_id]

            if tool:
                self.tools[tool_id] = tool

    def get_available_tools(self) -> List[str]:
        """Get list of available tools from tool service"""
        return self.tool_service.list_tools_with_parameters()

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

        tools_help = "Available Tools:\n"
        for tool_name in task_config.tools:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                tools_help += f"{tool.name}: {tool.description}\n"
                if hasattr(tool, "args_schema"):
                    schema = tool.args_schema.schema()
                    tools_help += (
                        f"Input: {json.dumps(schema.get('example', {}), indent=2)}\n\n"
                    )

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
                {tools_help}
            """

    async def create_task(
        self,
        query: str,
        node_ids: Optional[List[str]] = None,
        context: str = "",
        task_index: int = 0,
    ) -> Task:
        """Create a task with proper context and description"""
        if task_index >= len(self.config.tasks):
            raise ValueError(
                f"Task index {task_index} out of range. Only {len(self.config.tasks)} tasks available."
            )

        # Ensure agent is initialized
        if not self.agent:
            self.agent = await self.create_agent()

        task_config = self.config.tasks[task_index]
        task_description = self._create_task_description(
            task_config, query, node_ids, context
        )

        # Create task with agent and LLM
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="Markdown formatted response with code context and explanations",
            llm=self.llm,
        )

        logger.info(f"Created task {task_index + 1} with LLM configuration")
        return task

    async def run(
        self,
        agent_id: str,
        query: str,
        project_id: str,
        conversation_id: str,
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Main execution flow"""
        try:
            # Set project_id from run parameters or config
            self.project_id = project_id or self.config.project_id
            if not self.project_id:
                raise ValueError(
                    "project_id must be provided either in config or as a parameter"
                )

            # # Update project_id in all tools
            # for tool in self.tools.values():
            #     tool.project_id = self.project_id

            logger.info(f"Running agent with project_id: {self.project_id}")

            # Create agent
            self.agent = await self.create_agent()

            # Create all tasks
            tasks = []
            for i, task_config in enumerate(self.config.tasks):
                task = await self.create_task(query, node_ids, task_index=i)
                tasks.append(task)
                logger.info(f"Created task {i+1}/{len(self.config.tasks)}")

            # Create single crew with all tasks
            crew = Crew(
                agents=[self.agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

            logger.info(f"Starting Crew AI kickoff with {len(tasks)} tasks")
            result = await crew.kickoff_async()

            return {"response": result.raw, "conversation_id": conversation_id}

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def create_agent(self) -> Agent:
        """Create the main agent with tools and configuration"""
        agent = Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools=list(self.tools.values()),
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )
        logger.info("Created CrewAI Agent instance")
        return agent
