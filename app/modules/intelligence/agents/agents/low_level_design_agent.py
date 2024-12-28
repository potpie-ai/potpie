import asyncio
import os
from contextlib import redirect_stdout
from typing import AsyncGenerator, Dict, List

import aiofiles
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

# Import necessary tools (assuming they're available in your project)
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    get_code_file_structure_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    get_node_neighbours_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)


class DesignStep(BaseModel):
    step_number: int = Field(..., description="The order of the design step")
    description: str = Field(..., description="Description of the design step")
    relevant_files: List[str] = Field(
        ..., description="List of relevant files for this step"
    )
    code_changes: Dict[str, str] = Field(
        ..., description="Proposed code changes for each file"
    )


class LowLevelDesignPlan(BaseModel):
    feature_name: str = Field(..., description="Name of the feature being implemented")
    overview: str = Field(
        ..., description="High-level overview of the implementation plan"
    )
    design_steps: List[DesignStep] = Field(
        ..., description="Detailed steps for implementing the feature"
    )
    potential_challenges: List[str] = Field(
        ..., description="Potential challenges or considerations"
    )


class LowLevelDesignAgent:
    def __init__(self, sql_db, llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = int(os.getenv("MAX_ITER", 10))
        self.sql_db = sql_db
        self._llm = None  # Initialize as None
        self._llm_provider = ProviderService(sql_db, user_id)
        self.user_id = user_id

        # Initialize tools
        self.get_code_from_node_id = get_code_from_node_id_tool(sql_db, user_id)
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.ask_knowledge_graph_queries = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )
        self.get_code_file_structure = get_code_file_structure_tool(sql_db)
        self.get_node_neighbours_from_node_id = get_node_neighbours_from_node_id_tool(
            sql_db
        )

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.CREWAI)
        return self._llm

    async def create_agents(self):
        llm = await self._get_llm()  # Use the helper method
        codebase_analyst = Agent(
            role="Codebase Analyst",
            goal="Analyze the existing codebase and provide insights on the current structure and patterns",
            backstory="""You are an expert in analyzing complex codebases. Your task is to understand the
            current project structure, identify key components, and provide insights that will help in
            planning new feature implementations.""",
            tools=[
                self.get_nodes_from_tags,
                self.ask_knowledge_graph_queries,
                self.get_code_from_node_id,
                self.get_code_from_probable_node_name,
                self.get_code_file_structure,
            ],
            allow_delegation=False,
            verbose=True,
            llm=llm,
        )

        design_planner = Agent(
            role="Design Planner",
            goal="Create a detailed low-level design plan for implementing new features",
            backstory="""You are a senior software architect specializing in creating detailed,
            actionable design plans. Your expertise lies in breaking down complex features into
            manageable steps and providing clear guidance for implementation.""",
            tools=[
                self.get_nodes_from_tags,
                self.ask_knowledge_graph_queries,
                self.get_code_from_node_id,
                self.get_code_from_probable_node_name,
                self.get_code_file_structure,
                self.get_node_neighbours_from_node_id,
            ],
            allow_delegation=True,
            verbose=True,
            llm=llm,
        )

        return codebase_analyst, design_planner

    async def create_tasks(
        self,
        functional_requirements: str,
        project_id: str,
        codebase_analyst,
        design_planner,
    ):
        analyze_codebase_task = Task(
            description=f"""
            Analyze the existing codebase for repo id {project_id} to understand its structure and patterns.
            Focus on the following:
            1. Identify the main components and their relationships.
            2. Determine the current architecture and design patterns in use.
            3. Locate areas that might be affected by the new feature described in: {functional_requirements}
            4. Identify any existing similar features or functionality that could be leveraged.

            Use the provided tools to query the knowledge graph and retrieve relevant code snippets as needed.
            You can use the probable node name tool to get the code for a node by providing a partial file or function name.
            Provide a comprehensive analysis that will aid in creating a low-level design plan.
            """,
            agent=codebase_analyst,
            expected_output="Codebase analysis report with insights on project structure and patterns",
        )

        create_design_plan_task = Task(
            description=f"""

            Based on the codebase analysis of repo id {project_id} and the following functional requirements: {functional_requirements}
            Create a detailed low-level design plan for implementing the new feature. Your plan should include:
            1. A high-level overview of the implementation approach.
            2. Detailed steps for implementing the feature, including:
               - Specific files that need to be modified or created.
               - Proposed code changes or additions for each file.
               - Any new classes, methods, or functions that need to be implemented.
            3. Potential challenges or considerations for the implementation.
            4. Any suggestions for maintaining code consistency with the existing codebase.

            Use the provided tools to query the knowledge graph and retrieve or propose code snippets as needed.
            You can use the probable node name tool to get the code for a node by providing a partial file or function name.
            Ensure your output follows the structure defined in the LowLevelDesignPlan Pydantic model.
            """,
            agent=design_planner,
            context=[analyze_codebase_task],
            expected_output="Low-level design plan for implementing the new feature",
        )

        return [analyze_codebase_task, create_design_plan_task]

    async def run(
        self, functional_requirements: str, project_id: str
    ) -> AsyncGenerator[str, None]:
        codebase_analyst, design_planner = await self.create_agents()
        tasks = await self.create_tasks(
            functional_requirements, project_id, codebase_analyst, design_planner
        )

        read_fd, write_fd = os.pipe()

        async def kickoff():
            with os.fdopen(write_fd, "w", buffering=1) as write_file:
                with redirect_stdout(write_file):
                    crew = Crew(
                        agents=[codebase_analyst, design_planner],
                        tasks=tasks,
                        process=Process.sequential,
                        verbose=True,
                    )
                    await crew.kickoff_async()

        asyncio.create_task(kickoff())

        # Stream the output
        final_answer_streaming = False
        async with aiofiles.open(read_fd, mode="r") as read_file:
            async for line in read_file:
                if not line:
                    break
                if final_answer_streaming:
                    if line.endswith("\x1b[00m\n"):
                        yield line[:-6]
                    else:
                        yield line
                if "## Final Answer:" in line:
                    final_answer_streaming = True


async def create_low_level_design_agent(
    functional_requirements: str,
    project_id: str,
    sql_db,
    llm,
    user_id: str,
) -> AsyncGenerator[str, None]:
    provider_service = ProviderService(sql_db, user_id)
    crew_ai_llm = await provider_service.get_large_llm(agent_type=AgentType.CREWAI)
    design_agent = LowLevelDesignAgent(sql_db, crew_ai_llm, user_id)
    async for chunk in design_agent.run(functional_requirements, project_id):
        yield chunk
