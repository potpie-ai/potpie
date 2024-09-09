import os
from typing import List, Optional

import requests
from langchain.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from app.modules.parsing.knowledge_graph.inference_schema import (
    QueryRequest,
    QueryResponse,
)
from app.modules.conversations.message.message_schema import ContextNode

class KnowledgeGraphInput(BaseModel):
    query: str = Field(
        description="A natural language question to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )
    node_ids: List[str] = Field(
        description="A list of node IDs to narrow down the search context, in case there are none, pass an empty list"
    )

class TestPlanOutput(BaseModel):
    happy_path: List[str] = Field(
        description="A list of happy path test scenarios"
    )
    edge_cases: List[str] = Field(
        description="A list of edge case test scenarios"
    )


class CodeTools:

    @staticmethod
    def get_docstrings(project_id: str, node_ids: List[str]) -> str:
        """
        Get the docstrings for a given query
        """
        from neo4j import GraphDatabase


        return f"Docstrings for query: {query}"

    @staticmethod
    def test_plan_tool(query: str) -> str:
        """
        Test plan tool
        """
        prompt = f"""A good integration test suite should aim to:
            - Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of standard test packages to make the tests easy to write and maintain
- Be easy to read and understand, with clean descriptive names
- Be deterministic, so that the tests always pass or fail in the same way
Happy Path Scenarios:
- Test cases that cover the expected normal operation of the function, where the inputs are valid and the function produces the expected output without any errors.
Edge Case Scenarios:
- Test cases that explore the boundaries of the function's input domain, such as:
  * Boundary values for input parameters
  * Unexpected or invalid input types
  * Error conditions and exceptions
  * Interactions with external dependencies (e.g., databases, APIs)
- These scenarios test the function's robustness and error handling capabilities.
To help integration test the flow above:
1. Analyze the provided code and explanation.
3. List diverse happy path and edge case scenarios that the function should handle. 
4. Include exactly 3 scenario statements of happy paths and 3 scenarios of edge cases. 
5. Format your output in JSON format as such, each scenario is only a string statement:
"""
        return f"Test tool response for query: {query}"

    @staticmethod
    def ask_knowledge_graph(
        query: str, project_id: str = None, node_ids: List[str] = []
    ) -> List[QueryResponse]:
        """
        Query the code knowledge graph using natural language questions.
        The knowledge graph performs a vector search over a corpus of code converted to docstrings at different granularity levels across the codebase, including function, class, and file levels.
        Use this to answer questions about the codebase.
        Args:
            query (str): A natural language question to ask the knowledge graph.
            project_id (str, optional): The project id metadata for the project being evaluated.
            node_ids (List[str], optional): A list of node IDs to narrow down the search context.

        Returns:
            List[QueryResponse]: A list of query responses containing relevant code snippets and their metadata.
        """
        data = QueryRequest(
            project_id=project_id, query=query, node_ids=node_ids
        ).model_dump()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('INTERNAL_CALL_SECRET')}",
        }
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL") + "api/v1/query"
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(kg_query_url, json=data, headers=headers)
        response = requests.post(kg_query_url, json=data, headers=headers)
        return [QueryResponse(**item) for item in response.json()]

    @staticmethod
    def _prepare_node_ids(node_ids: Optional[List[ContextNode]]) -> List[str]:
        return [node.node_id for node in node_ids] if node_ids else []

    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            Tool(
                name="search_code",
                func=lambda query, project_id, node_ids: CodeTools.search_code(
                    query, project_id, CodeTools._prepare_node_ids(node_ids)
                ),
                description="Search for relevant code snippets",
            ),
            Tool(
                name="get_file_content",
                func=lambda query, project_id, node_ids: CodeTools.get_file_content(
                    query, project_id, CodeTools._prepare_node_ids(node_ids)
                ),
                description="Get the content of a specific file",
            ),
            StructuredTool.from_function(
                func=cls.ask_knowledge_graph,
                name="Ask Knowledge Graph",
                description="Query the code knowledge graph with specific directed questions about the codebase using natural language. Do not use this to query code directly.",
                args_schema=KnowledgeGraphInput,
            ),
        ]
