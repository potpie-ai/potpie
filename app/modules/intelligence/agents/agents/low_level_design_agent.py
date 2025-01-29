import asyncio
import os
from contextlib import redirect_stdout
from typing import AsyncGenerator, Dict, List

import aiofiles
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

# Import necessary tools (assuming they're available in your project)
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts_provider.agent_prompts_provider import (
    AgentPromptsProvider,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.prompt_schema import PromptType
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
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
        self.llm = llm
        self.user_id = user_id
        self.prompt_service = PromptService(self.sql_db)


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

    async def create_agents(self):
        
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        codebase_analyst_prompt = await self.prompt_service.get_prompts(
            "codebase_analyst",
            [PromptType.SYSTEM],
            preferred_llm,
            max_iter=self.max_iter,
        )
        codebase_analyst = Agent(
            role=codebase_analyst_prompt["role"],
            goal=codebase_analyst_prompt["goal"],
            backstory=codebase_analyst_prompt["backstory"],
            tools=[
                self.get_nodes_from_tags,
                self.ask_knowledge_graph_queries,
                self.get_code_from_node_id,
                self.get_code_from_probable_node_name,
                self.get_code_file_structure,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        design_planner_prompt = await self.prompt_service.get_prompts(
        "design_planner",
        [PromptType.SYSTEM],
        preferred_llm,
        max_iter=self.max_iter,
        )
        
        design_planner = Agent(
            role=design_planner_prompt["role"],
            goal=design_planner_prompt["goal"],
            backstory=design_planner_prompt["backstory"],
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
            llm=self.llm,
        )

        return codebase_analyst, design_planner

    async def create_tasks(
        self,
        functional_requirements: str,
        project_id: str,
        codebase_analyst,
        design_planner,
    ):  
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        analyze_task_prompt = await self.prompt_service.get_prompts(
            "analyze_codebase_task",
            [PromptType.SYSTEM],
            preferred_llm,
            project_id=project_id,
            functional_requirements=functional_requirements,
            max_iter=self.max_iter,
        )
        
        analyze_codebase_task = Task(
            description=analyze_task_prompt,
            agent=codebase_analyst,
            expected_output="Codebase analysis report with insights on project structure and patterns",
        )

        design_task_prompt = await self.prompt_service.get_prompts(
            "create_design_plan_task",
            [PromptType.SYSTEM],
            preferred_llm,
            project_id=project_id,
            functional_requirements=functional_requirements,
            max_iter=self.max_iter,
            LowLevelDesignPlan= self.LowLevelDesignPlan,
        )
        create_design_plan_task = Task(
            description=design_task_prompt,
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
    provider_service = LLMProviderService(sql_db, user_id)
    crew_ai_llm = provider_service.get_large_llm(agent_type=AgentLLMType.CREWAI)
    design_agent = LowLevelDesignAgent(sql_db, crew_ai_llm, user_id)
    async for chunk in design_agent.run(functional_requirements, project_id):
        yield chunk
