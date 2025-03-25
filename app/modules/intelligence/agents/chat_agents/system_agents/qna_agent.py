from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator, List
from pydantic import BaseModel
import json


class NodeContext(BaseModel):
    node_id: str
    name: str


class QnAAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider

    def _build_agent(self) -> ChatAgent:
        # The agent configuration now focuses on generating test files with precise implementation patterns
        agent_config = AgentConfig(
            role="Senior Test Automation Engineer",
            goal=(
                "Extract detailed implementation patterns from project code to generate production-ready functional tests that "
                "exactly match the project's existing structure, utility usage, JSON handling, mock configuration, and assertion patterns."
            ),
            backstory="""
You are a senior test automation engineer with years of expertise in this specific project's codebase. 
You have a photographic memory of the project's test architecture, utility classes, file organization, and coding standards.
You focus first on extracting concrete implementation details from the provided code, identifying specific patterns for:
1. Test class structure, package names, and import statements
2. JSON file locations and loading mechanisms using project-specific utilities
3. External service mocking with the exact helper methods used in the project
4. Assertion patterns using the project's custom assertor classes
5. Test lifecycle management including setup/teardown with the project's established methods

You never make assumptions about implementation patterns - you extract them directly from the provided code examples.
You generate complete, production-ready test files that could be committed directly to the codebase without modifications.
            """,
            tasks=[
                TaskConfig(
                    description="""
## Task 1: Extract Test Implementation Patterns

Analyze the provided code nodes to create a concrete implementation guide for generating tests in this project. Your output must include:

### 1. Test Structure Specification
- Exact package naming convention with examples from the code (e.g., `package com.swiggy.api.functional.cart`)
- Complete import statement pattern with all required classes
- Test class naming pattern with concrete examples
- Test method naming pattern with concrete examples

### 2. JSON Data Management Specification
- Extract exact paths to JSON test files from the code (e.g., `src/functionalTest/resources/requestJson/cartUpdate/`)
- Identify specific JSON file names used for similar test cases
- Extract the exact code pattern used to load and parse JSON files, including:
  - The utility classes used (with import statements)
  - The method calls for loading and deserializing JSON
  - The object types JSON is parsed into

### 3. Mock Configuration Specification
- Extract the exact initialization pattern for WireMock and gRPC stubs
- Identify helper classes and methods used for specific mock configurations
- Extract concrete examples of mock setup code for services related to the functionality under test
- Identify patterns for different mock response types (success, error, timeout)

### 4. Assertion Pattern Specification
- Identify assertor classes used for the functionality under test
- Extract concrete assertion method calls with parameters
- Document assertion hierarchy and organization

### 5. Test Lifecycle Management Specification
- Extract exact @BeforeClass and @AfterClass method implementations
- Identify helper methods called during setup (e.g., commonUOMSSetup())
- Extract pattern for test cleanup and resource management

### 6. Business Logic Analysis
- Document the key functionality of the component under test
- Identify input parameters, expected outputs, and error conditions
- Map the component's interactions with other services

Your output must include direct code examples extracted from the provided code nodes for each section, showing the exact patterns used in the project. Do not provide general guidelines - extract and document the specific implementation patterns from the code.
                    """,
                    expected_output="Comprehensive implementation pattern guide with concrete code examples extracted from the project for test structure, JSON handling, mocking, assertions, and lifecycle management."
                ),
                TaskConfig(
                    description="""
## Task 2: Generate Production-Ready Test Implementation

Using your implementation pattern guide, create a complete, production-ready test file for the functionality under test. Your implementation must:

1. Follow the exact package structure identified in Task 1
2. Include all necessary import statements based on extracted patterns
3. Implement the correct test class structure with proper naming
4. Include proper test lifecycle methods (@BeforeClass, @AfterClass) with exactly the helper methods identified in Task 1
5. Load test data using the exact JSON file paths and utility methods identified in the project
6. Configure mocks using the exact helper methods and patterns identified in the project
7. Implement test methods covering:
   - Happy path scenario
   - Error handling scenarios
   - Edge cases
8. Use the exact assertion patterns and assertor classes identified in the project
9. Implement proper cleanup using the exact patterns identified in the project

Your output must be a complete, compilable Java test file that exactly matches the project's structure and patterns. The file should be ready to commit to the codebase without any modifications.

Be extremely precise with:
- Package declaration
- Import statements
- Class structure and naming
- Method implementations
- JSON file paths and loading
- Mock configuration
- Test assertions
- Documentation style and formatting

Ensure your implementation pattern matches exactly what was observed in the project code. The generated test must be virtually indistinguishable from a test written by an engineer familiar with the project's codebase.
                    """,
                    expected_output="Complete, production-ready integration test file that exactly matches the project's structure and patterns, ready for immediate use without modifications."
                ),
            ],
        )
        tools = self.tools_provider.get_tools([
            "get_code_from_multiple_node_ids",
            "get_code_from_probable_node_name",
            "ask_knowledge_graph_queries",
            "get_code_file_structure",
            "intelligent_code_graph",
            "think",
        ])
        return CrewAIAgent(self.llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            # Retrieve graphs for each node to understand component relationships, using the intelligent code graph tool
            graphs = {}
            all_node_contexts: List[NodeContext] = []
            for node_id in ctx.node_ids:
                if not node_id:
                    continue  # Skip empty node_ids
                    
                try:
                    # Use intelligent_code_graph with named parameters in a dictionary
                    graph = self.tools_provider.tools["intelligent_code_graph"].run({
                        "project_id": ctx.project_id, 
                        "node_id": node_id, 
                        "relevance_threshold": 0.7,  # Only include highly relevant nodes
                        "max_depth": 5  # Reasonable depth for full context
                    })
                    
                    # Check if there was an error
                    if graph and "error" in graph:
                        ctx.additional_context += f"Error processing node {node_id}: {graph['error']}\n"
                        continue
                        
                    graphs[node_id] = graph

                    def extract_unique_node_contexts(node, visited=None):
                        if visited is None:
                            visited = set()
                        node_contexts: List[NodeContext] = []
                        if node["id"] not in visited:
                            visited.add(node["id"])
                            node_contexts.append(NodeContext(node_id=node["id"], name=node["name"]))
                            for child in node.get("children", []):
                                node_contexts.extend(extract_unique_node_contexts(child, visited))
                        return node_contexts

                    node_contexts = extract_unique_node_contexts(graph["graph"]["root_node"])
                    all_node_contexts.extend(node_contexts)
                except Exception as e:
                    ctx.additional_context += f"Error processing node {node_id}: {str(e)}\n"

            if not graphs:
                ctx.additional_context += "Unable to retrieve any code graphs. Please check the provided node IDs.\n"
                return ctx

            seen = set()
            unique_node_contexts: List[NodeContext] = []
            for node in all_node_contexts:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    unique_node_contexts.append(node)

            formatted_graphs = {}
            for node_id, graph in graphs.items():
                formatted_graphs[node_id] = {
                    "name": next((node.name for node in unique_node_contexts if node.node_id == node_id), "Unknown"),
                    "structure": graph["graph"]["root_node"],
                }

            ctx.additional_context += f"- Code structure and component relationships from INTELLIGENT knowledge graph (noise filtered):\n{json.dumps(formatted_graphs, indent=2)}\n"

            # Add summary of filtering
            total_evaluated = sum(graph["graph"].get("nodes_evaluated", 0) for graph in graphs.values())
            total_included = sum(graph["graph"].get("nodes_included", 0) for graph in graphs.values())
            if total_evaluated > 0:
                ctx.additional_context += f"- The intelligent context builder evaluated {total_evaluated} nodes and included only the {total_included} most relevant ones ({(total_included/total_evaluated)*100:.1f}%), filtering out noise.\n\n"

            expanded_node_ids = [node.node_id for node in unique_node_contexts]
            if expanded_node_ids:
                code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(ctx.project_id, expanded_node_ids)
                ctx.additional_context += f"Code for all RELEVANT components in the integration flow:\n{code_results}\n"
                ctx.node_ids = expanded_node_ids
            else:
                ctx.additional_context += "No relevant code components were found after filtering.\n"

        # Identify test data structure
        json_structure = await self.tools_provider.tools["get_code_file_structure"].arun({
            "project_id": ctx.project_id,
            "path": "src/functionalTest/resources"
        })
        ctx.additional_context += f"Test data structure in the project:\n{json_structure}\n"

        # Original context building - Get file structure
        file_structure = await self.tools_provider.file_structure_tool.fetch_repo_structure(ctx.project_id)
        ctx.additional_context += f"File Structure of the project:\n{file_structure}\n"

        test_related_paths = [
            path for path in file_structure.split('\n')
            if any(test_term in path.lower() for test_term in ["test", "mock", "stub", "assert", "functional", "integration"])
        ]
        if test_related_paths:
            ctx.additional_context += "Test-related paths identified:\n" + "\n".join(test_related_paths) + "\n"

        ctx.additional_context += (
            "\nImportant: Generated tests must reuse JSON templates from src/functionalTest/resources/requestJson and src/functionalTest/resources/responseJson (please fetch file structure for these directories to get more detailed json file names)"
            "configure mocks using existing helper methods (e.g., SetWiremockStub, GripMockDataStubbing), and utilize centralized assertor classes for validations."
        )

        ctx.additional_context += (
            "\nComprehensive Functional Test Writing Guidelines:\n"
            "1. Test Data Setup: Store JSON templates in structured directories; use getObjectMapper().readValue() for deserialization.\n"
            "2. External Service Mocking: Configure mocks with WireMock and gRPC stub helpers; simulate behaviors and reset mocks as required.\n"
            "3. Assertions & Validations: Use assertor classes (e.g., CartUpdateAssertor) for detailed, field-level checks.\n"
            "4. Test Lifecycle: Employ @BeforeClass and @AfterClass with helper methods like commonUOMSSetup and commonUOMSCleanup.\n"
            "5. Test Organization: Group tests by feature in designated directories with consistent naming conventions.\n"
        )

        ctx.additional_context += (
            "\nIntegration Test Specific Guidelines:\n"
            "1. Component Relationships: Understand the complete flow between components from the code graph.\n"
            "2. Data Flow: Trace data transformations through the integration chain.\n"
            "3. Boundary Testing: Validate interfaces between components for correct data handling.\n"
            "4. Error Propagation: Verify that errors propagate and are handled as expected.\n"
        )

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_ctx = await self._enriched_context(ctx)
        return await self._build_agent().run(enriched_ctx)

    async def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(enriched_ctx):
            yield chunk