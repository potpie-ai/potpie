import os
from typing import Dict, List, Any
import json

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    get_code_from_multiple_node_ids_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.web_tools.github_tool import github_tool
from app.modules.intelligence.tools.web_tools.webpage_extractor_tool import (
    webpage_extractor_tool,
)


class IntegrationTestAgent:
    def __init__(self, sql_db, llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iterations = os.getenv("MAX_ITER", 15)
        self.sql_db = sql_db
        self.llm = llm
        self.user_id = user_id
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        if os.getenv("FIRECRAWL_API_KEY"):
            self.webpage_extractor_tool = webpage_extractor_tool(sql_db, user_id)
        if os.getenv("GITHUB_APP_ID"):
            self.github_tool = github_tool(sql_db, user_id)

    async def create_agents(self):
        tools = (
            [
                self.get_code_from_probable_node_name,
                self.get_code_from_multiple_node_ids,
            ]
            + (
                [self.webpage_extractor_tool]
                if hasattr(self, "webpage_extractor_tool")
                else []
            )
            + ([self.github_tool] if hasattr(self, "github_tool") else [])
        )

        integration_test_agent = Agent(
            role="Integration Test Writer",
            goal="Create a comprehensive integration test suite for the provided codebase",
            backstory="You are an expert in writing integration tests for code using latest features of the popular testing libraries for the given programming language.",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=tools,
            max_iter=self.max_iterations,
        )

        return integration_test_agent

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
        graphs: Dict[str, Dict[str, Any]],
        history: List[str],
        integration_test_agent,
    ):
        node_ids_list = [node.node_id for node in node_ids]

        # Format graphs for better readability in the prompt
        formatted_graphs = {}
        for node_id, graph in graphs.items():
            formatted_graphs[node_id] = {
                "name": next(
                    (node.name for node in node_ids if node.node_id == node_id),
                    "Unknown",
                ),
                "structure": graph["graph"]["root_node"],
            }

        integration_test_task = Task(
            description=f"""Your mission is to create comprehensive test plans and corresponding integration tests based on the user's query and provided code.
            Given the following context:
            - Chat History: {history}

            Process:
            1. **Code Graph Analysis:**
            - Code structure is defined in multiple graphs for each component:
            {json.dumps(formatted_graphs, indent=2)}
            - **Graph Structure:**
                - Analyze each provided graph structure to understand the code flow and component interactions.
                - Identify all major components, their dependencies, and interaction points.
                - Pay special attention to how different components interact across different graphs.
            - **Code Retrieval:**
                - Fetch the docstrings and code for the provided node IDs using the `Get Code and docstring From Multiple Node IDs` tool.
                - Node IDs: {', '.join(node_ids_list)}
                - Project ID: {project_id}
                - Fetch the code for all relevant nodes in each graph to understand the full context of the codebase.

            2. **Analysis:**
            - Analyze the fetched code and docstrings to understand the functionality.
            - Identify the purpose, inputs, outputs, and potential side effects of each component.
            - Understand how components from different graphs interact with each other.

            3. **Decision Making:**
            - Refer to the chat history to determine if a test plan or integration tests have already been generated.
            - If a test plan exists and the user requests modifications or additions, proceed accordingly without regenerating the entire plan.
            - If no existing test plan or integration tests are found, generate new ones based on the user's query.

            4. **Test Plan Generation:**
            Generate a test plan only if a test plan is not already present in the chat history or the user asks for it again.
            - For each component and their interactions, create detailed test plans covering:
                - Happy path scenarios
                - Edge cases (e.g., empty inputs, maximum values, type mismatches)
                - Error handling
                - Any relevant performance or security considerations
                - Cross-component interactions and their edge cases
            - Format the test plan in two sections "Happy Path" and "Edge Cases" as neat bullet points

            5. **Integration Test Writing:**
            - Write complete integration tests based on the test plans.
            - Use appropriate testing frameworks and best practices.
            - Include clear, descriptive test names and explanatory comments.
            - Ensure tests cover interactions between components from different graphs.

            6. **Reflection and Iteration:**
            - Review the test plans and integration tests.
            - Ensure comprehensive coverage and correctness.
            - Make refinements as necessary, respecting the max iterations limit of {self.max_iterations}.

            7. **Response Construction:**
            - Provide the test plans and integration tests in your response.
            - Include any necessary explanations and notes.
            - Ensure the response is clear and well-organized.

            Constraints:
            - Refer to the user's query: "{query}"
            - Consider the chat history for any specific instructions or context.
            - Respect the max iterations limit of {self.max_iterations} when planning and executing tools.

            Ensure that your final response is JSON serializable and follows the specified pydantic model: {self.TestAgentResponse.model_json_schema()}
            Don't wrap it in ```json or ```python or ```code or ```
            For citations, include only the file_path of the nodes fetched and used.
            """,
            expected_output="Write COMPLETE CODE for integration tests for each node based on the test plan.",
            agent=integration_test_agent,
            output_pydantic=self.TestAgentResponse,
            async_execution=True,
        )

        return integration_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        chat_history: List,
    ) -> Dict[str, str]:
        integration_test_agent = await self.create_agents()

        # Get graphs for each node to understand component relationships
        graphs = {}
        all_node_contexts = []

        for node in node_ids:
            # Get the code graph for each node
            graph = GetCodeGraphFromNodeIdTool(self.sql_db).run(
                project_id, node.node_id
            )
            graphs[node.node_id] = graph

            def extract_unique_node_contexts(node, visited=None):
                if visited is None:
                    visited = set()
                node_contexts = []
                if node["id"] not in visited:
                    visited.add(node["id"])
                    node_contexts.append(
                        NodeContext(node_id=node["id"], name=node["name"])
                    )
                    for child in node.get("children", []):
                        node_contexts.extend(
                            extract_unique_node_contexts(child, visited)
                        )
                return node_contexts

            # Extract related nodes from each graph
            node_contexts = extract_unique_node_contexts(graph["graph"]["root_node"])
            all_node_contexts.extend(node_contexts)

        # Remove duplicates while preserving order
        seen = set()
        unique_node_contexts = []
        for ctx in all_node_contexts:
            if ctx.node_id not in seen:
                seen.add(ctx.node_id)
                unique_node_contexts.append(ctx)

        integration_test_task = await self.create_tasks(
            unique_node_contexts,
            project_id,
            query,
            graphs,
            chat_history,
            integration_test_agent,
        )

        crew = Crew(
            agents=[integration_test_agent],
            tasks=[integration_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_integration_test_agent(
    query: str,
    chat_history: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    user_id: str,
) -> Dict[str, str]:
    if not node_ids:
        return {
            "error": "No function name is provided by the user. The agent cannot generate test plan or test code without specific class or function being selected by the user. Request the user to use the '@ followed by file or function name' feature to link individual functions to the message. "
        }

    # Get graphs for each node to understand component relationships
    graphs = {}
    all_node_contexts = []

    for node in node_ids:
        # Get the code graph for each node
        graph = GetCodeGraphFromNodeIdTool(sql_db).run(project_id, node.node_id)
        graphs[node.node_id] = graph

        def extract_unique_node_contexts(node, visited=None):
            if visited is None:
                visited = set()
            node_contexts = []
            if node["id"] not in visited:
                visited.add(node["id"])
                node_contexts.append(NodeContext(node_id=node["id"], name=node["name"]))
                for child in node.get("children", []):
                    node_contexts.extend(extract_unique_node_contexts(child, visited))
            return node_contexts

        # Extract related nodes from each graph
        node_contexts = extract_unique_node_contexts(graph["graph"]["root_node"])
        all_node_contexts.extend(node_contexts)

    # Remove duplicates while preserving order
    seen = set()
    unique_node_contexts = []
    for ctx in all_node_contexts:
        if ctx.node_id not in seen:
            seen.add(ctx.node_id)
            unique_node_contexts.append(ctx)

    integration_test_agent = IntegrationTestAgent(sql_db, llm, user_id)
    result = await integration_test_agent.run(
        project_id, unique_node_contexts, query, chat_history
    )
    return result
