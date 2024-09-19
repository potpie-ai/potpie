import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.crewai_agents.test_plan_agent import TestPlanAgent
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_tools,
)


class UnitTestAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.code_tools = get_code_tools(self.sql_db)
        self.test_plan_agent = TestPlanAgent(sql_db, llm)
        self.llm = llm

    async def create_agents(self):
        test_plan_agent = await self.test_plan_agent.create_agents()

        unit_test_agent = Agent(
            role="Unit Test Writer",
            goal="Write unit tests based on test plans",
            backstory="You are a Unit Test Virtuoso, crafting elegant and robust test suites that stand the test of time. You are an expert in using all popular testing libraries for the given programming language. Your tests are not just code; they're a form of documentation and a safety net for future development.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return test_plan_agent, unit_test_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the Markdown formatted test plan and the test suite code block",
        )
        citations: List[str] = Field(
            ..., description="Exhaustive List of file names referenced in the response"
        )

    async def create_tasks(
        self,
        node_ids: List[NodeContext],
        project_id: str,
        query: str,
        history: List,
        test_plan_agent,
        unit_test_agent,
    ):
        fetch_docstring_task, test_plan_task = await self.test_plan_agent.create_tasks(
            node_ids, project_id, query, test_plan_agent
        )

        unit_test_task = Task(
            description=f"""Write unit tests corresponding on the test plans. Closely refer the provided code for the functions to generate accurate unit test code.
            Your mission:
            Transform the provided test plans into a comprehensive suite of unit tests, ensuring that every scenario is covered with precision and clarity.

            Process:
            1. Test Setup:
            - Import necessary testing frameworks and modules
            - Review the provided test plan and the fetched node details from previous tasks.
            - Identify any additional classes/functions required for mocking:
               a. Use the get_code_from_probable_node_name tool to fetch its code if not in the provided node IDs. The probable node names look like "filename:class_name" or "filename:function_name"
               b. Validate the result of the get_code_from_probable_node_name tool against the probable node name. Discard from context if it does not match. 
            - Set up any required test fixtures or mocks
            - Refer the code context closely to write accurate tests.

            2. Test Writing:
            For each scenario in the test plan:
            - Write a descriptive test name (e.g., "test_valid_discount_calculation")
            - Implement the test using the Arrange-Act-Assert pattern
            - Add clear comments explaining the purpose and expectations of each test
            - Refer to the {query} and {history[-min(5, len(history)):]} for any specific instructions and follow them.
            - Write COMPLETE code for the unit tests.

            3. Edge Case and Error Handling:
            - Implement tests for all identified edge cases
            - Write tests to verify proper error handling and exception raising

            4. Code Coverage:
            - Ensure your tests cover all branches and paths in the code
            - Use code coverage tools if available to verify comprehensive coverage

            5. Readability and Maintainability:
            - Use consistent naming conventions
            - Extract common setup into helper methods or fixtures
            - Keep tests focused and avoid unnecessary complexity

            6. Reflection:
            - Review your test suite
            - Ask yourself: "Would another developer understand the purpose and expectations of each test?"
            - Refine and improve based on your reflection

            Project Id: {project_id}
            For citations, include only the file_path of the nodes fetched and used.
            Ensure that your final response is JSON serialisable but dont wrap it in ```json or ```python or ```code or ```
            Ensure that your output follows the following pydantic model: {self.TestAgentResponse.model_json_schema()}""",
            expected_output=f"Outline the test plan and write unit tests for each node based on the test plan.",
            agent=unit_test_agent,
            context=[fetch_docstring_task, test_plan_task],
            output_pydantic=self.TestAgentResponse,
            tools=[self.code_tools[2]]
        )

        return fetch_docstring_task, test_plan_task, unit_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        chat_history: List,
    ) -> Dict[str, str]:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        test_plan_agent, unit_test_agent = await self.create_agents()
        docstring_task, test_plan_task, unit_test_task = await self.create_tasks(
            node_ids, project_id, query, chat_history, test_plan_agent, unit_test_agent
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
    query: str,
    chat_history: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
) -> Dict[str, str]:
    unit_test_agent = UnitTestAgent(sql_db, llm)
    result = await unit_test_agent.run(project_id, node_ids, query, chat_history)
    return result
