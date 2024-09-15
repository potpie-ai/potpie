import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field
from app.modules.intelligence.tools.kg_based_tools.code_tools import CodeTools
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.change_detection.change_detection import ChangeDetectionResponse, get_blast_radius_tool


class BlastRadiusAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.llm = llm

    async def create_agents(self):

        blast_radius_agent = Agent(
            role="Blast Radius Agent",
            goal="Explain the blast radius of the changes made in the code.",
            backstory="You are an expert in understanding the impact of code changes on the codebase.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return blast_radius_agent

    class BlastRadiusAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response describing the blast radius of the changes made in the code.",
        )
        citations: List[str] = Field(
            ..., description="List of file names extracted from contest and referenced in the response"
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

            You also have access the the query knowledge graph tool. Use it to answer natural language questions about the codebase during the analysis.

            Analyze the changes fetched and explain their blast radius. Consider the following:
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
            expected_output=f"Structured output outlining the impact of the code changes on the codebase and any other query by the user following the pydantic schema; {self.BlastRadiusAgentResponse.model_json_schema()}",
            agent=blast_radius_agent,
            tools=[get_blast_radius_tool()[0],CodeTools.get_tools()[0]],
            output_pydantic=self.BlastRadiusAgentResponse
        )

        return analyze_changes_task

    async def run(
        self, project_id: str, node_ids: List[NodeContext], query: str
    ) -> Dict[str, str]:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

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


async def kickoff_blast_radius_crew(
    query: str, project_id: str, node_ids: List[NodeContext], sql_db, llm
) -> Dict[str, str]:
    blast_radius_agent = BlastRadiusAgent(sql_db, llm)
    result = await blast_radius_agent.run(project_id, node_ids, query)
    return result
