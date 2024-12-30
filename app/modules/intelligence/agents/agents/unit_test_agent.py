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
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)


class UnitTestAgent:
    def __init__(self, sql_db, llm, user_id):
        self.sql_db = sql_db
        self.user_id = user_id
        self._llm = None  # Initialize as None
        self._llm_provider = ProviderService(sql_db, user_id)
        self.get_code_from_node_id = get_code_from_node_id_tool(sql_db, user_id)
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.max_iterations = os.getenv("MAX_ITER", 15)

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.CREWAI)
        return self._llm

    async def create_agents(self):
        llm = await self._get_llm()  # Use the helper method
        unit_test_agent = Agent(
            role="Test Plan and Unit Test Expert",
            goal="Create test plans and write unit tests based on user requirements",
            backstory="You are a seasoned AI test engineer specializing in creating robust test plans and unit tests. You aim to assist users effectively in generating and refining test plans and unit tests, ensuring they are comprehensive and tailored to the user's project requirements.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
            max_iter=self.max_iterations,
        )

        return unit_test_agent

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
        unit_test_agent,
    ):
        node_ids_list = [node.node_id for node in node_ids]

        unit_test_task = Task(
            description=f"""Your mission is to create comprehensive test plans and corresponding unit tests based on the user's query and provided code.
            Given the following context:
            - Chat History: {history}

            Process:
            1. **Code Retrieval:**
            - If not already present in the history, Fetch the docstrings and code for the provided node IDs using the get_code_from_node_id tool.
            - Node IDs: {', '.join(node_ids_list)}
            - Project ID: {project_id}
            - Fetch the code for the file path of the function/class mentioned in the user's query using the get code from probable node name tool. This is needed for correct inport of class name in the unit test file.

            2. **Analysis:**
            - Analyze the fetched code and docstrings to understand the functionality.
            - Identify the purpose, inputs, outputs, and potential side effects of each function/method.

            3. **Decision Making:**
            - Refer to the chat history to determine if a test plan or unit tests have already been generated.
            - If a test plan exists and the user requests modifications or additions, proceed accordingly without regenerating the entire plan.
            - If no existing test plan or unit tests are found, generate new ones based on the user's query.

            4. **Test Plan Generation:**
            Generate a test plan only if a test plan is not already present in the chat history or the user asks for it again.
            - For each function/method, create a detailed test plan covering:
                - Happy path scenarios
                - Edge cases (e.g., empty inputs, maximum values, type mismatches)
                - Error handling
                - Any relevant performance or security considerations
            - Format the test plan in two sections "Happy Path" and "Edge Cases" as neat bullet points

            5. **Unit Test Writing:**
            - Write complete unit tests based on the test plans.
            - Use appropriate testing frameworks and best practices.
            - Include clear, descriptive test names and explanatory comments.

            6. **Reflection and Iteration:**
            - Review the test plans and unit tests.
            - Ensure comprehensive coverage and correctness.
            - Make refinements as necessary, respecting the max iterations limit of {self.max_iterations}.

            7. **Response Construction:**
            - Provide the test plans and unit tests in your response.
            - Include any necessary explanations or notes.
            - Ensure the response is clear and well-organized.

            Constraints:
            - Refer to the user's query: "{query}"
            - Consider the chat history for any specific instructions or context.
            - Respect the max iterations limit of {self.max_iterations} when planning and executing tools.

            Ensure that your final response is JSON serializable and follows the specified pydantic model: {self.TestAgentResponse.model_json_schema()}
            Don't wrap it in ```json or ```python or ```code or ```
            For citations, include only the file_path of the nodes fetched and used.
            """,
            expected_output="Outline the test plan and write unit tests for each node based on the test plan.",
            agent=unit_test_agent,
            output_pydantic=self.TestAgentResponse,
            tools=[self.get_code_from_probable_node_name, self.get_code_from_node_id],
            async_execution=True,
        )

        return unit_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        chat_history: List,
    ) -> Dict[str, str]:
        unit_test_agent = await self.create_agents()
        unit_test_task = await self.create_tasks(
            node_ids, project_id, query, chat_history, unit_test_agent
        )

        crew = Crew(
            agents=[unit_test_agent],
            tasks=[unit_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()

        return result


async def kickoff_unit_test_agent(
    query: str,
    chat_history: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    user_id,
) -> Dict[str, str]:
    if not node_ids:
        return {
            "error": "No function name is provided by the user. The agent cannot generate test plan or test code without specific class or function being selected by the user. Request the user to use the '@ followed by file or function name' feature to link individual functions to the message. "
        }
    provider_service = ProviderService(sql_db, user_id)
    crew_ai_llm = await provider_service.get_large_llm(agent_type=AgentType.CREWAI)
    unit_test_agent = UnitTestAgent(sql_db, crew_ai_llm, user_id)
    result = await unit_test_agent.run(project_id, node_ids, query, chat_history)
    return result
