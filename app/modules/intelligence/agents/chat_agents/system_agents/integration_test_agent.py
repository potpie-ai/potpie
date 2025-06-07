import json
from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.classification_prompts import AgentType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator, List
from pydantic import BaseModel


class NodeContext(BaseModel):
    node_id: str
    name: str


class IntegrationTestAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Integration Test Writer",
            goal="Create a comprehensive integration test suite for the provided codebase. Analyze the code, determine the appropriate testing language and framework, and write tests that cover all major integration points.",
            backstory="""
                    You are an expert in writing unit tests for code using latest features of the popular testing libraries for the given programming language.
                """,
            tasks=[
                TaskConfig(
                    description=integration_test_task_prompt,
                    expected_output="Write COMPLETE CODE for integration tests for each node based on the test plan.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_code_from_probable_node_name",
                "webpage_extractor",
                "web_search_tool",
                "github_tool",
                "fetch_file",
                "analyze_code_structure",
            ]
        )

        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            return AdaptiveAgent(
                llm_provider=self.llm_provider,
                prompt_provider=self.prompt_provider,
                rag_agent=CrewAIAgent(self.llm_provider, agent_config, tools),
                agent_type=AgentType.INTEGRATION_TEST,
            )

    def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if not ctx.node_ids or len(ctx.node_ids) == 0:
            return ctx

        # Get graphs for each node to understand component relationships
        graphs = {}
        all_node_contexts: List[NodeContext] = []

        for node_id in ctx.node_ids:
            # Get the code graph for each node
            graph = self.tools_provider.get_code_graph_from_node_id_tool.run(
                ctx.project_id, node_id
            )
            graphs[node_id] = graph

            def extract_unique_node_contexts(node, visited=None):
                if visited is None:
                    visited = set()
                node_contexts: List[NodeContext] = []
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
        unique_node_contexts: List[NodeContext] = []
        for node in all_node_contexts:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_node_contexts.append(node)

        # Format graphs for better readability in the prompt
        formatted_graphs = {}
        for node_id, graph in graphs.items():
            formatted_graphs[node_id] = {
                "name": next(
                    (
                        node.name
                        for node in unique_node_contexts
                        if node.node_id == node_id
                    ),
                    "Unknown",
                ),
                "structure": graph["graph"]["root_node"],
            }

        ctx.additional_context += f"- Code structure is defined in multiple graphs for each component: \n{json.dumps(formatted_graphs, indent=2)}"
        ctx.node_ids = [node.node_id for node in unique_node_contexts]

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


integration_test_task_prompt = """

    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution
    HOW TO GUIDE:

    IMPORATANT: steps on HOW TO traverse the codebase:
    1. You can use websearch, docstrings, readme to understand current feature/code you are working with better. Understand how to use current feature in context of codebase
    2. Use AskKnowledgeGraphQueries tool to understand where perticular feature or functionality resides or to fetch specific code related to some keywords. Fetch file structure to understand the codebase better, Use FetchFile tool to fetch code from a file
    3. Use GetcodefromProbableNodeIDs tool to fetch code for perticular class or function in a file, Use analyze_code_structure to get all the class/function/nodes in a file
    4. Use GetcodeFromMultipleNodeIDs to fetch code for nodeIDs fetched from tools before
    5. Use GetNodeNeighboursFromNodeIDs to fetch all the code referencing current code or code referenced in the current node (code snippet)
    6. Above tools and steps can help you figure out full context about the current code in question
    7. Figure out how all the code ties together to implement current functionality
    8. Fetch Dir structure of the repo and use fetch file tool to fetch entire files, if file is too big the tool will throw error, then use code analysis tool to target proper line numbers (feel free to use set startline and endline such that few extra context lines are also fetched, tool won't throw out of bounds exception and return lines if they exist)
    9. Use above mentioned tools to fetch imported code, referenced code, helper functions, classes etc to understand the control flow

    Your mission is to create comprehensive test plans and corresponding integration tests based on the user's query and provided code.

    **Process:**

    **Detailed Component Analysis:**
    - **Functionality Understanding:**
        - For each component identified in the graph, analyze its purpose, inputs, outputs, and potential side effects.
        - Understand how each component interacts with others within the system.
    - **Import Resolution:**
        - Determine the necessary imports for each component by analyzing the graph structure.
        - Use the `get_code_from_probable_node_name` tool to fetch code snippets for accurate import statements.
        - Validate that the fetched code matches the expected component names and discard any mismatches.

    **Test Plan Generation:**
    Generate a test plan only if a test plan is not already present in the chat history or the user asks for it again.
    - **Comprehensive Coverage:**
        - For each component and their interactions, create detailed test plans covering:
        - **Happy Path Scenarios:** Typical use cases where interactions work as expected.
        - **Edge Cases:** Scenarios such as empty inputs, maximum values, type mismatches, etc.
        - **Error Handling:** Cases where components should handle errors gracefully.
        - **Performance Considerations:** Any relevant performance or security aspects that should be tested.
    - **Integration Points:**
        - Identify all major integration points between components that require testing to ensure seamless interactions.
    - Format the test plan in two sections "Happy Path" and "Edge Cases" as neat bullet points.

    **Integration Test Writing:**
    - **Test Suite Development:**
        - Based on the generated test plans, write comprehensive integration tests that cover all identified scenarios and integration points.
        - Ensure that the tests include:
        - **Setup and Teardown Procedures:** Proper initialization and cleanup for each test to maintain isolation.
        - **Mocking External Dependencies:** Use mocks or stubs for external services and dependencies to isolate the components under test.
        - **Accurate Imports:** Utilize the analyzed graph structure to include correct import statements for all components involved in the tests.
        - **Descriptive Test Names:** Clear and descriptive names that explain the scenario being tested.
        - **Assertions:** Appropriate assertions to validate expected outcomes.
        - **Comments:** Explanatory comments for complex test logic or setup.

    **Reflection and Iteration:**
    - **Review and Refinement:**
        - Review the test plans and integration tests to ensure comprehensive coverage and correctness.
        - Make refinements as necessary, respecting the max iterations limit.

"""
