import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.crewai_agents.test_plan_agent import TestPlanAgent
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_tool,
)


class UnitTestAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.get_code_tool = get_tool(self.sql_db)
        self.test_plan_agent = TestPlanAgent(sql_db, llm)
        self.llm = llm

    async def create_agents(self):
        test_plan_agent = await self.test_plan_agent.create_agents()

        unit_test_agent = Agent(
            role="Unit Test Writer",
            goal="Write unit tests based on test plans",
            backstory="You are an expert in writing unit tests for code using latest features of the popular testing libraries for the given programming language.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return test_plan_agent, unit_test_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the test plan and the test suite",
        )
        citations: List[str] = Field(
            ..., description="List of file names referenced in the response"
        )

    async def create_tasks(
        self,
        node_ids: List[NodeContext],
        project_id: str,
        query: str,
        test_plan_agent,
        unit_test_agent,
    ):
        fetch_docstring_task, test_plan_task = await self.test_plan_agent.create_tasks(
            node_ids, project_id, query, test_plan_agent
        )

        unit_test_task = Task(
            description=f"""Write unit tests corresponding on the test plans. Closely refer the provided code for the functions to generate accurate unit test code.
            Refer the {query} for any specific instructions and follow them.
A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of popular testing libraries for that language (unless a library is specifically mentioned) to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way""",
            expected_output=f"Outline the test plan and write unit tests for each node based on the test plan. Write complete code for the unit tests. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model :\n{self.TestAgentResponse.model_json_schema()}",
            agent=unit_test_agent,
            context=[fetch_docstring_task, test_plan_task],
            output_pydantic=self.TestAgentResponse,
        )

        return fetch_docstring_task, test_plan_task, unit_test_task

    async def run(
        self, project_id: str, node_ids: List[NodeContext], query: str
    ) -> Dict[str, str]:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        test_plan_agent, unit_test_agent = await self.create_agents()
        docstring_task, test_plan_task, unit_test_task = await self.create_tasks(
            node_ids, project_id, query, test_plan_agent, unit_test_agent
        )

        crew = Crew(
            agents=[test_plan_agent, unit_test_agent],
            tasks=[docstring_task, test_plan_task, unit_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()

        return result


async def kickoff_unit_test_crew(
    query: str, project_id: str, node_ids: List[NodeContext], sql_db, llm
) -> Dict[str, str]:
    unit_test_agent = UnitTestAgent(sql_db, llm)
    result = await unit_test_agent.run(project_id, node_ids, query)
    return result
