import json
import logging
from typing import Any, Dict, List, Optional, Union

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, validator

from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.key_management.secret_manager import SecretManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)


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
                logging.error(f"Error parsing expected_output: {str(e)}")
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


class CustomAgent:
    def __init__(self, agent_config: Dict[str, Any], secret_manager: SecretManager):
        """Initialize the agent with configuration"""
        self.config = AgentConfig(**agent_config)
        self.secret_manager = secret_manager
        self.max_iter = 5

        self.user_id = self.config.user_id
        self.role = self.config.role
        self.goal = self.config.goal
        self.backstory = self.config.backstory
        self.llm = ProviderService(self.db, self.user_id).get_large_llm(
            AgentType.CREWAI
        )

        self.llm = None
        self.agent = None

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
                Node IDs: {node_ids}
                Additional Context: {context}

                TASK:
                {task_config.description}

                Expected Output Format:
                {json.dumps(task_config.expected_output, indent=2)}

                INSTRUCTIONS:
                1. Process and synthesize the gathered information
                2. Format your response in markdown
                3. Include relevant code snippets and file references
                4. Provide clear explanations

                IMPORTANT:
                - You have a maximum of 10 iterations unless specified otherwise
                - Use tools efficiently and avoid unnecessary API calls
            """

    def create_task(
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

        # Ensure agent and LLM are initialized
        if not self.agent:
            self.agent = self.create_agent()

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

        logging.info(f"Created task {task_index + 1} with LLM configuration")
        return task

    async def run(self, query: str, node_ids: Optional[Any] = None) -> Any:
        """Main execution flow"""
        # Create agent
        self.agent = self.create_agent()

        # Create all tasks
        tasks = []
        for i, task_config in enumerate(self.config.tasks):
            task = self.create_task(query, node_ids, task_index=i)
            tasks.append(task)
            logging.info(f"Created task {i+1}/{len(self.config.tasks)}")

        # Create single crew with all tasks
        crew = Crew(
            agents=[self.agent], tasks=tasks, process=Process.sequential, verbose=True
        )

        logging.info(f"Starting Crew AI kickoff with {len(tasks)} tasks")
        result = await crew.kickoff_async()

        logging.info(f"Final result: {result}")
        return result

    def create_agent(self) -> Agent:
        """Create the main agent with tools and configuration"""

        agent = Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )
        logging.info("Created CrewAI Agent instance")
        return agent
