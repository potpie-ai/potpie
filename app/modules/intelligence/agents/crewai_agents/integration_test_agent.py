import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.crewai_agents.test_plan_agent import TestPlanAgent
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_tools,
)


class IntegrationTestAgent:
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.get_code_tool = get_code_tools(self.sql_db)
        self.test_plan_agent = TestPlanAgent(sql_db, llm)
        self.llm = llm

    async def create_agents(self):
        test_plan_agent = await self.test_plan_agent.create_agents()

        integration_test_agent = Agent(
            role="Integration Test Writer",
            goal="Create a comprehensive integration test suite for the provided codebase. Analyze the code, determine the appropriate testing language and framework, and write tests that cover all major integration points.",
            backstory="You are an expert in writing unit tests for code using latest features of the popular testing libraries for the given programming language.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return test_plan_agent, integration_test_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the test plan and the test suite",
        )
        citations: List[str] = Field(
            ..., description="Exhaustive List of file names referenced in the response"
        )

    async def create_tasks(
        self,
        node_ids: List[NodeContext],
        project_id: str,
        query: str,
        graph: Dict[str, Any],
        history: List[str],
        test_plan_agent,
        integration_test_agent,
    ):
        fetch_docstring_task, test_plan_task = await self.test_plan_agent.create_tasks(
            node_ids, project_id, query, test_plan_agent
        )

        integration_test_task = Task(
            description=f"""
            1. Analyze the provided codebase:
            - Code structure is defined in the {graph}
            - Determine the programming language used
            - Identify the main components and their interactions
            - Refer the query: '{query}'\n and the history: \n'{history[-min(5, len(history)):]}' for any specific instructions and follow them.
            - Review the provided test plan and the fetched node details from previous tasks.
            - Identify any additional classes/functions required for mocking:
                a. Use the get_code_from_probable_node_name tool to fetch its code if not in the provided node IDs. The probable node names look like "filename:class_name" or "filename:function_name"
                b. Validate the result of the get_code_from_probable_node_name tool against the probable node name. Discard from context if it does not match. 
            - Set up any required test fixtures or mocks
            - Refer the code context closely to write accurate tests.


            2. Choose an appropriate testing framework based on the language used in the codebase.

            3. Write a comprehensive integration test suite that follows the test plan and includes:
            - Setup and teardown procedures for each test
            - Mocking of external services and dependencies
            - Tests for all major integration points between components
            - relevant imports. If you don't know the exact import, DO NOT GUESS, use a placeholder and mention it to the user.

            4. Follow these best practices:
            - Use descriptive test names that explain the scenario being tested
            - Group related tests together
            - Ensure each test is independent and can run in isolation
            - Use appropriate assertions to validate expected outcomes
            - Include comments explaining complex test logic or setup

            5.Output:
            - A complete integration test suite in the appropriate language and framework
            - Setup and teardown procedures
            - Mocking implementations for external dependencies

            Additional Notes:
            - If the codebase uses a specific testing framework or follows certain testing conventions, adhere to those.
            - Consider performance implications and optimize tests where possible without sacrificing coverage.
            - If you encounter any ambiguities or need more information about the codebase, request clarification before proceeding.

            Remember, the goal is to create a robust, maintainable, and comprehensive integration test suite that will help ensure the reliability and correctness of the system's component interactions.
            Ensure that your final response is JSON serialisable but dont wrap it in ```json or ```python or ```code or ```""",
            expected_output=f"Write COMPLETE CODE for integration tests for each node based on the test plan. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model:\n{self.TestAgentResponse.model_json_schema()}",
            agent=integration_test_agent,
            context=[fetch_docstring_task, test_plan_task],
            output_pydantic=self.TestAgentResponse
        )

        return fetch_docstring_task, test_plan_task, integration_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        graph: Dict[str, Any],
        history: List,
    ) -> Dict[str, str]:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        test_plan_agent, integration_test_agent = await self.create_agents()
        docstring_task, test_plan_task, integration_test_task = await self.create_tasks(
            node_ids, project_id, query, graph, history, test_plan_agent, integration_test_agent
        )

        crew = Crew(
            agents=[test_plan_agent, integration_test_agent],
            tasks=[docstring_task, test_plan_task, integration_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_integration_test_crew(
    query: str, project_id: str, node_ids: List[NodeContext], sql_db, llm, history: List[str]
) -> Dict[str, str]:
    graph = GetCodeGraphFromNodeIdTool(sql_db).run(project_id, node_ids[0].node_id)

    def extract_node_ids(node):
        node_ids = []
        for child in node.get("children", []):
            node_ids.extend(extract_node_ids(child))
        return node_ids

    def extract_unique_node_contexts(node, visited=None):
        if visited is None:
            visited = set()
        node_contexts = []
        if node["id"] not in visited:
            visited.add(node["id"])
            node_contexts.append(
                NodeContext(node_id=node["id"], name=node["name"])
            )  # Assuming NodeContext can be initialized with node data
            for child in node.get("children", []):
                node_contexts.extend(extract_unique_node_contexts(child, visited))
        return node_contexts

    node_contexts = extract_unique_node_contexts(graph["graph"]["root_node"])
    integration_test_agent = IntegrationTestAgent(sql_db, llm)
    result = await integration_test_agent.run(project_id, node_contexts, query, graph, history)
    return result
