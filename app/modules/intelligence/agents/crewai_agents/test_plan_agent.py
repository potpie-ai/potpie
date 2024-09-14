import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_tool,
)


class TestPlanAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.get_code_tool = get_tool(self.sql_db)
        self.llm = llm

    async def create_agents(self):
        test_plan_agent = Agent(
            role="Test Plan Creator",
            goal="Fetch docstrings for given node IDs. Create test plans with happy paths and edge cases based on understanding of docstrings",
            backstory="You excel at creating comprehensive test plans by analyzing code intent.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return test_plan_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the test plan and the test suite",
        )
        citations: List[str] = Field(
            ..., description="List of file names referenced in the response"
        )

    async def create_tasks(
        self, node_ids: List[NodeContext], project_id: str, query: str, test_plan_agent
    ):
        node_ids_list = [node.node_id for node in node_ids]
        fetch_docstring_task = Task(
            description=f"Fetch docstrings and code for the following node IDs: {', '.join(node_ids_list)} for Project id {project_id}",
            expected_output="A dictionary mapping node IDs to their docstring and code",
            agent=test_plan_agent,
            tools=self.get_code_tool,
        )

        test_plan_task = Task(
            description="Given the docstrings and code for given node_ids, create test plans with happy paths and edge cases for each node based on the docstrings",
            expected_output="A dictionary mapping node IDs to their test plans (happy paths and edge cases)",
            agent=test_plan_agent,
            context=[fetch_docstring_task],
        )

        return fetch_docstring_task, test_plan_task

    async def run(
        self, project_id: str, node_ids: List[NodeContext], query: str
    ) -> Dict[str, str]:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        test_plan_agent = await self.create_agents()
        docstring_task, test_plan_task = await self.create_tasks(
            node_ids, project_id, query, test_plan_agent
        )

        crew = Crew(
            agents=[test_plan_agent],
            tasks=[docstring_task, test_plan_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()

        return result


async def kickoff_unit_test_crew(
    query: str, project_id: str, node_ids: List[NodeContext], sql_db
) -> Dict[str, str]:
    test_plan_agent = TestPlanAgent(sql_db)
    result = await test_plan_agent.run(project_id, node_ids, query)
    return result
