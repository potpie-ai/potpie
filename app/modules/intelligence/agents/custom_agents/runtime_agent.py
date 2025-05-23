import json
import os
from typing import Any, Dict, List, Optional, Union

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.provider.provider_service import (
    AgentProvider,
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
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
        db: Session,
        agent_config: Dict[str, Any],
    ):
        """Initialize the agent with configuration and tools"""
        self.config = AgentConfig(**agent_config)
        self.db = db
        self.max_iter = int(os.getenv("MAX_ITER", "5"))

        self.user_id = self.config.user_id
        self.role = self.config.role
        self.goal = self.config.goal
        self.backstory = self.config.backstory
        self.history_manager = ChatHistoryService(self.db)
        self.project_id = None
        self.agent = None
        self.llm = ProviderService(self.db, self.user_id).get_llm(
            AgentProvider.CREWAI, config_type="chat"
        )
        # Initialize tools
        self.tool_service = ToolService(db, self.user_id)
        self.tools = {}
        self.agent_tools = set()
        for task_config in self.config.tasks:
            self.agent_tools.update(task_config.tools)
        for tool_id, tool in self.get_available_tools().items():
            if tool_id in self.agent_tools:
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


        return f"""
        You are an advanced AI coding agent specializing in comprehensive task completion. Your role is to analyze the given context, plan efficiently, and execute tasks using the available tools and information.

        First, review the project's file structure:

        <additional_context>
        {context}
        </additional_context>

        You have access to the following tools:

        <tools_help>
        {tools_help}
        </tools_help>

        Now, consider the user's prompt:

        <user_prompt>
        {query}
        </user_prompt>

        Here is the relevant project information:

        <project_id>{self.project_id}</project_id>

        <node_ids>{node_ids}</node_ids>

        Your task is as follows:

        <task_description>
        {task_config.description}
        </task_description>

        You have a maximum of <max_iterations>{self.max_iter}</max_iterations> iterations to complete this task.

        Before proceeding with the task, conduct a thorough project analysis inside <project_analysis> tags. Consider the following:
        1. Analyze the project structure and identify key components.
        2. Break down the user's prompt and extract main objectives.
        3. List out the available tools and their potential applications for this task.
        4. Prioritize subtasks based on their importance and dependencies.
        5. Identify the most efficient way to use the provided tools.
        6. Plan how to avoid redundant tool calls and maximize efficiency.
        7. Determine the best order of operations to complete the task.
        8. Consider potential challenges, edge cases, and how to overcome them.

        After your analysis phase, follow these instructions:

        1. Execute your plan, using the tools as necessary.
        2. Process and synthesize all gathered information.
        3. Format your response in markdown, including relevant code snippets and file references.
        4. Provide clear explanations for each step of your process.
        5. Ensure your response is comprehensive yet concise.

        Remember to think critically throughout the process and justify your decisions. Efficient use of resources and clear communication are key to successfully completing this task.

            """

    async def create_task(
        self,
        query: str,
        node_ids: Optional[List[str]] = None,
        context: str = "",
        task_index: int = 0,
        previous_task: Optional[Task] = None,
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
            expected_output=f"{task_config.expected_output}",
            context=[previous_task] if previous_task else None,
        )

        logger.info(f"Created task {task_index + 1} with LLM configuration")
        return task

    async def run(
        self,
        agent_id: str,
        query: str,
        project_id: str,
        project_name: str,
        conversation_id: str,
        node_ids: Optional[List[str]] = None,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main execution flow"""
        try:
            # Set project_id from run parameters or config
            self.project_id = project_id or self.config.project_id
            if not self.project_id:
                raise ValueError(
                    "project_id must be provided either in config or as a parameter"
                )

            logger.info(f"Running agent with project_id: {self.project_id}")
            # file_structure = (
            #     await self.tool_service.file_structure_tool.fetch_repo_structure(
            #         self.project_id
            #     )
            # )
            # context = f"File Structure of the project:\n {file_structure}"
            context = await self.tool_service.process_large_pr_tool.arun({"project_id": self.project_id, "base_branch": "prev-3"})
            # context = ""
            # Create all tasks with context chaining
            tasks = []
            previous_task = None
            for i, task_config in enumerate(self.config.tasks):
                task = await self.create_task(
                    query,
                    node_ids,
                    f"Project Name (github: owner/repo): {project_name}\n{context}",  # Empty context string since we're using task context
                    task_index=i,
                    previous_task=previous_task,
                )
                tasks.append(task)
                previous_task = task
                logger.info(f"Created task {i+1}/{len(self.config.tasks)}")


            crew = Crew(
                agents=[self.agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

            logger.info(f"Starting Crew AI kickoff with {len(tasks)} tasks")
            result = await crew.kickoff_async()
            content = result.raw
            self.history_manager.add_message_chunk(
                conversation_id,
                content,
                MessageType.AI_GENERATED,
            )
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

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
