import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts_provider.agent_prompts_provider import (
    AgentPromptsProvider,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.prompt_schema import PromptType
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
from app.modules.intelligence.tools.change_detection.change_detection_tool import (
    ChangeDetectionResponse,
    get_change_detection_tool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)


class BlastRadiusAgent:
    def __init__(self, sql_db, user_id, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.user_id = user_id
        self.llm = llm
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.ask_knowledge_graph_queries = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )
        self.prompt_service = PromptService(self.sql_db)


    async def create_agents(self):
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        agent_prompt = await self.prompt_service.get_prompts(
            "blast_radius_agent",
            [PromptType.SYSTEM],
            preferred_llm,
        )
        
        blast_radius_agent = Agent(
            role=agent_prompt["role"],
            goal=agent_prompt["goal"],
            backstory=agent_prompt["backstory"],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return blast_radius_agent

    class BlastRadiusAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response describing the analysis of the changes made in the code.",
        )
        citations: List[str] = Field(
            ...,
            description="List of file names extracted from context and referenced in the response",
        )

    async def create_tasks(
        self,
        project_id: str,
        query: str,
        blast_radius_agent,
    ):
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        task_prompt = await self.prompt_service.get_prompts(
            "analyze_changes_task",
            [PromptType.SYSTEM],
            preferred_llm,
            project_id=project_id,
            query=query,
            ChangeDetectionResponse=ChangeDetectionResponse,
            BlastRadiusAgentResponse=self.BlastRadiusAgentResponse,
        )
        

        analyze_changes_task = Task(
            description=task_prompt,
            expected_output=f"Comprehensive impact analysis of the code changes on the codebase and answers to the users query about them. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model : {self.BlastRadiusAgentResponse.model_json_schema()}",
            agent=blast_radius_agent,
            output_pydantic=self.BlastRadiusAgentResponse,
            async_execution=True,
        )

        return analyze_changes_task

    async def run(
        self, project_id: str, node_ids: List[NodeContext], query: str
    ) -> Dict[str, str]:
        blast_radius_agent = await self.create_agents()
        blast_radius_task = await self.create_tasks(
            project_id, query, blast_radius_agent
        )

        crew = Crew(
            agents=[blast_radius_agent],
            tasks=[blast_radius_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()

        return result


async def kickoff_blast_radius_agent(
    query: str, project_id: str, node_ids: List[NodeContext], sql_db, user_id, llm
) -> Dict[str, str]:
    provider_service = LLMProviderService(sql_db, user_id)
    crew_ai_mini_llm = provider_service.get_small_llm(agent_type=AgentLLMType.CREWAI)
    blast_radius_agent = BlastRadiusAgent(sql_db, user_id, crew_ai_mini_llm)
    result = await blast_radius_agent.run(project_id, node_ids, query)
    return result
