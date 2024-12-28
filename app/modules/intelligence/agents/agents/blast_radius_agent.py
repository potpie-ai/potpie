import os
import asyncio
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
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
        self._llm = None  # Initialize as None
        self._llm_provider = ProviderService(sql_db, user_id)
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.ask_knowledge_graph_queries = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.CREWAI)
        return self._llm

    async def create_agents(self):
        llm = await self._get_llm()  # Use the helper method
        blast_radius_agent = Agent(
            role="Blast Radius Agent",
            goal="Explain the blast radius of the changes made in the code.",
            backstory="You are an expert in understanding the impact of code changes on the codebase.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
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
        analyze_changes_task = Task(
            description=f"""Fetch the changes in the current branch for project {project_id} using the get code changes tool.
            The response of the fetch changes tool is in the following format:
            {ChangeDetectionResponse.model_json_schema()}
            In the response, the patches contain the file patches for the changes.
            The changes contain the list of changes with the updated and entry point code. Entry point corresponds to the API/Consumer upstream of the function that the change was made in.
            The citations contain the list of file names referenced in the changed code and entry point code.

            You also have access the the query knowledge graph tool to answer natural language questions about the codebase during the analysis.
            Based on the response from the get code changes tool, formulate queries to ask details about specific changed code elements.
            1. Frame your query for the knowledge graph tool:
            - Identify key concepts, code elements, and implied relationships from the changed code.
            - Consider the context from the users query: {query}.
            - Determine the intent and key technical terms.
            - Transform into keyword phrases that might match docstrings:
                * Use concise, functionality-based phrases (e.g., "creates document MongoDB collection").
                * Focus on verb-based keywords (e.g., "create", "define", "calculate").
                * Include docstring-related keywords like "parameters", "returns", "raises" when relevant.
                * Preserve key technical terms from the original query.
                * Generate multiple keyword variations to increase matching chances.
                * Be specific in keywords to improve match accuracy.
                * Ensure the query includes relevant details and follows a similar structure to enhance similarity search results.

            2. Execute your formulated query using the knowledge graph tool.

            Analyze the changes fetched and explain their impact on the codebase. Consider the following:
            1. Which functions or classes have been directly modified?
            2. What are the potential side effects of these changes?
            3. Are there any dependencies that might be affected?
            4. How might these changes impact the overall system behavior?
            5. Based on the entry point code, determine which APIs or consumers etc are impacted by the changes.

            Refer to the {query} for any specific instructions and follow them.

            Based on the analysis, provide a structured inference of the blast radius:
            1. Summarize the direct changes
            2. List potential indirect effects
            3. Identify any critical areas that require careful testing
            4. Suggest any necessary refactoring or additional changes to mitigate risks
            6. If the changes are impacting multiple APIs/Consumers, then say so.


            Ensure that your output ALWAYS follows the structure outlined in the following pydantic model:
            {self.BlastRadiusAgentResponse.model_json_schema()}""",
            expected_output=f"Comprehensive impact analysis of the code changes on the codebase and answers to the users query about them. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model : {self.BlastRadiusAgentResponse.model_json_schema()}",
            agent=blast_radius_agent,
            tools=[
                get_change_detection_tool(self.user_id),
                self.get_nodes_from_tags,
                self.ask_knowledge_graph_queries,
            ],
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
    provider_service = ProviderService(sql_db, user_id)
    crew_ai_mini_llm = await provider_service.get_small_llm(agent_type=AgentType.CREWAI)
    blast_radius_agent = BlastRadiusAgent(sql_db, user_id, crew_ai_mini_llm)
    result = await blast_radius_agent.run(project_id, node_ids, query)
    return result
