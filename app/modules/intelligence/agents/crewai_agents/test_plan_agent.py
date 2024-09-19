import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_tools,
)


class TestPlanAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.code_tools = get_code_tools(self.sql_db)
        self.llm = llm

    async def create_agents(self):
        test_plan_agent = Agent(
            role="Test Plan Creation Expert",
            goal="Fetch docstrings and code for given node IDs. Understand the code under test using the fetched code and its docstrings. Finally create test plans with comprehensive happy paths and edge cases based on understanding of the flow and user's requirements",
            backstory="You are a world-class Test Plan Architect with a keen eye for detail and a knack for anticipating edge cases. Your mission is to create comprehensive test plans that serve as a blueprint for bulletproof software.",
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
            description=f"""
            1. Analyze the query: "{query}"
            2. Fetch docstrings and code ONLY FOR THE FOLLOWING NODE IDs using the get_code_from_node_id tool: {', '.join(node_ids_list)} for Project id {project_id}
            3. For Project ID: {project_id}

            Final output:
            Provide a dictionary mapping node IDs to their docstring and code, including both initially provided and newly discovered relevant code.
            """,
            expected_output="A dictionary mapping node IDs to their docstring and code",
            agent=test_plan_agent,
            tools=[self.code_tools[0], self.code_tools[1]],
        )

        test_plan_task = Task(
            description="""You are a world-class Test Plan Architect with a keen eye for detail and a knack for anticipating edge cases. Your mission is to create comprehensive test plans that serve as a blueprint for bulletproof software.

            Process:
            1. Code Analysis:
            - Thoroughly examine the provided docstrings and code
            - Identify the purpose, inputs, outputs, and potential side effects of each function/method

            2. Test Scenario Identification:
            - For each function/method, list:
                a) Happy path scenarios
                b) Edge cases (e.g., empty inputs, maximum values, type mismatches)

            3. Test Plan Creation:
            - For each scenario, specify:
                a) Input data
                b) Expected output or behavior

            4. Reflection:
            - Review your test plan
            - Ask yourself: "What scenarios am I missing? What could go wrong that I haven't considered?"
            - Refine and expand the plan based on your reflection

            Provide a detailed test plan for each function/method, following this structured approach.""",
            expected_output=f"Outline the test plan including happy paths and edge cases for each node.",
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
