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
        agent_config = AgentConfig(
            role="Senior Test Automation Engineer",
            goal=(
                "Analyze code changes and existing tests to generate comprehensive test plans and implementations that maintain "
                "test coverage while adhering to the project's established testing patterns and standards."
            ),
            backstory="""
You are a senior test automation engineer with deep expertise in test planning and implementation. Your approach is methodical:
1. You first analyze code changes to understand what's been modified
2. You examine existing tests to understand current coverage and patterns
3. You create precise test plans that maintain or improve coverage
4. You implement tests following the project's exact patterns and standards

You have a systematic mind that:
- Thinks step-by-step through complex testing scenarios
- Identifies edge cases and boundary conditions
- Maintains consistency with existing test patterns
- Ensures complete test coverage of code changes
            """,
            tasks=[
                TaskConfig(
                    description="""
<task>Analyze Changes and Gather Test Context</task>

<objective>Build a comprehensive understanding of code changes and existing test patterns to enable precise test implementation.</objective>

<context>
You must gather all necessary information to write complete, production-ready tests:
- Identify all modified functions and their entry points
- Locate and analyze existing test files for similar components
- Extract exact test implementation patterns
- Map all dependencies and interactions
</context>

<steps>
1. Code Change Analysis
   - Use change detection to identify modified functions and their entry points
   - Analyze the nature of changes (new features, modifications, bug fixes)
   - Map dependencies and potential impact areas
   - Extract concrete examples of modified code patterns

2. Existing Test Discovery
   - Search for test files corresponding to changed components
   - Extract exact test implementation patterns:
     * Package and import statements
     * Test class structure and naming
     * Test method organization
     * Test data loading mechanisms
     * Mock configuration setup
     * Assertion patterns and helper methods
     * Test lifecycle management (@BeforeClass, @AfterClass)

3. Test Resource Analysis
   - Map JSON test data locations and patterns:
     * Exact file paths (e.g., src/functionalTest/resources/requestJson/)
     * File naming conventions
     * Data structure patterns
   - Document mock configuration patterns:
     * WireMock/gRPC initialization code
     * Helper methods and utilities
     * Response type patterns
   - Identify assertion utilities:
     * Assertion classes and methods
     * Validation patterns
     * Custom assertors

4. Test Coverage Analysis
   - Identify existing tests that need updates
   - Map areas requiring new tests
   - Document test dependencies and shared utilities
   - Note any tests that need to be removed

                    """,
                    expected_output="""Complete analysis of code changes and test patterns with concrete implementation examples from the project in the following format: \n <output_format>
{
  "code_changes": {
    "modified_functions": [{
      "name": str,
      "type": "new|modified|deleted",
      "code": str,
      "dependencies": [str]
    }],
    "entry_points": [{
      "name": str,
      "code": str,
      "callers": [str]
    }]
  },
  "test_patterns": {
    "package_structure": {
      "base_package": str,
      "imports": [str],
      "example_test_class": str
    },
    "test_resources": {
      "json_paths": {
        "request_json": str,
        "response_json": str
      },
      "example_files": [str]
    },
    "mock_setup": {
      "initialization_code": str,
      "helper_methods": [{
        "name": str,
        "code": str,
        "usage": str
      }],
      "example_usage": str
    },
    "assertions": {
      "assertor_classes": [{
        "name": str,
        "methods": [str],
        "example_usage": str
      }]
    },
    "lifecycle": {
      "setup_code": str,
      "cleanup_code": str,
      "helper_utilities": [str]
    }
  },
  "existing_tests": {
    "related_files": [{
      "path": str,
      "coverage": str,
      "patterns_used": [str]
    }],
    "to_update": [{
      "path": str,
      "reason": str
    }],
    "to_create": [{
      "component": str,
      "patterns_to_use": [str]
    }]
  }
}
</output_format> """
                ),
                TaskConfig(
                    description="""
<task>Generate Complete Test Implementation</task>

<objective>Create production-ready test files that exactly match the project's patterns and provide comprehensive coverage of code changes.</objective>

<context>
Using the analysis from Task 1, implement complete test files that:
- Follow exact package structure and naming conventions
- Include all necessary imports and dependencies
- Use correct test data and mock configurations
- Implement proper lifecycle management
- Provide comprehensive test coverage
</context>

<steps>
1. Test File Implementation
   - Create complete test files with proper package and imports
   - Implement test class with correct naming and structure
   - Add all necessary test lifecycle methods
   - Implement test data loading and mock configuration
   - Write comprehensive test methods for all scenarios

2. Test Method Implementation
   - Implement happy path test cases
   - Add error scenario tests
   - Cover edge cases and boundary conditions
   - Use correct assertion patterns
   - Add proper documentation

3. Resource Implementation
   - Create/update JSON test data files
   - Configure mock responses
   - Set up test utilities as needed
   - Implement cleanup handlers

4. Documentation
   - Add class and method documentation
   - Document test scenarios and coverage
   - Include setup instructions if needed
   - Document any new patterns or utilities

The output must be complete, compilable test files that match the project's patterns exactly. Include all necessary files, resources, and configurations.

                    """,
                    expected_output="""Complete, production-ready test files that exactly match the project's patterns and can be committed without modifications. Follow the following output format: \n <output_format>
// The complete test file implementation
package com.example.test;

import ...;

/**
 * Test class documentation
 */
@Test
public class ExampleTest {
    // Complete test implementation
    // Including:
    // - All imports
    // - Class structure
    // - Setup/teardown
    // - Test methods
    // - Assertions
    // - Documentation
}

// Additional resource files if needed:
// - JSON test data
// - Mock configurations
// - Helper utilities
</output_format>"""
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
            "change_detection",
        ])
        return CrewAIAgent(self.llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        # First, get code changes if project_id is available
        if ctx.project_id:
            try:
                changes = await self.tools_provider.tools["change_detection"].arun(ctx.project_id)
                if changes and changes.changes:
                    ctx.additional_context += "\nCode Changes Detected:\n"
                    for change in changes.changes:
                        ctx.additional_context += f"\nModified Code:\n{change.updated_code}\n"
                        ctx.additional_context += f"Entry Point:\n{change.entrypoint_code}\n"
                        ctx.additional_context += f"Affected Files:\n{', '.join(change.citations)}\n"
                    ctx.additional_context += "\nCode Patches:\n"
                    for file_path, patch in changes.patches.items():
                        ctx.additional_context += f"\nFile: {file_path}\n"
                        ctx.additional_context += f"Patch:\n{patch}\n"
                    
                    # Add the changed files to node_ids for further processing
                    if not ctx.node_ids:
                        ctx.node_ids = []
                    for change in changes.changes:
                        for citation in change.citations:
                            if citation not in ctx.node_ids:
                                ctx.node_ids.append(citation)
            except Exception as e:
                ctx.additional_context += f"\nError detecting code changes: {str(e)}\n"

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