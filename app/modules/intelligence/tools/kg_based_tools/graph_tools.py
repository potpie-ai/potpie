import asyncio
import os
from typing import List, Dict, Tuple

import aiohttp
import requests
from langchain.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from app.core.config_provider import ConfigProvider
from app.core.database import get_db
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService


class QueryRequest(BaseModel):
    node_ids: List[str] = Field(description="A list of node ids to query")
    project_id: str = Field(description="The project id metadata for the project being evaluated")
    query: str = Field(description="A natural language question to ask the knowledge graph")

class KnowledgeGraphInput(BaseModel):
    query: str = Field(
        description="A natural language question to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class GetNodesFromTagsInput(BaseModel):
    tags: List[str] = Field(description="A list of tags to filter the nodes by")
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class MultipleKnowledgeGraphQueriesInput(BaseModel):
    queries: List[str] = Field(
        description="A list of natural language questions to ask the knowledge graph"
    )
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class CodeTools:


    @staticmethod
    def get_nodes_from_tags(tags: List[str], project_id: str) -> str:
        """
        Get nodes from the knowledge graph based on the provided tags.
        Inputs for the get_nodes_from_tags method:
        - tags (List[str]): A list of tags to filter the nodes by.
           * API: Does the code define any API endpoint? Look for route definitions, request handling, or API client usage.
           * WEBSOCKET: Does the code implement or use WebSocket connections? Check for WebSocket-specific libraries or protocols.
           * PRODUCER: Does the code generate and send messages to a queue or topic? Look for message publishing or event emission.
           * CONSUMER: Does the code receive and process messages from a queue or topic? Check for message subscription or event handling.
           * DATABASE: Does the code interact with a database? Look for query execution, data insertion, updates, or deletions.
           * SCHEMA: Does the code define any database schema? Look for ORM models, table definitions, or schema-related code.
           * HTTP: Does the code make HTTP requests to external services? Check for HTTP client usage or request handling.
        - project_id (str): The ID of the project being evaluated, this is a UUID.
        """

        tag_conditions = " OR ".join([f"'{tag}' IN n.tags" for tag in tags])
        query = f"""MATCH (n:NODE)
        WHERE ({tag_conditions}) AND n.repoId = '{project_id}'
        RETURN n.file_path AS file_path, n.docstring AS docstring, n.text AS text, n.node_id AS node_id, n.name AS name
        """
        neo4j_config = ConfigProvider().get_neo4j_config()
        nodes = CodeGraphService(
            neo4j_config["uri"],
            neo4j_config["username"],
            neo4j_config["password"],
            next(get_db()),
        ).query_graph(query)
        return nodes

    @staticmethod
    async def ask_multiple_knowledge_graph_queries(queries: List[QueryRequest]) -> Dict[str, str]:
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL")
        headers = {"Content-Type": "application/json"}

        async def fetch_query(query_request: QueryRequest) -> Tuple[str, str]:
            data = query_request.dict()
            async with aiohttp.ClientSession() as session:
                async with session.post(kg_query_url, json=data, headers=headers) as response:
                    result = await response.json()
                    return query_request.query, result

        tasks = [fetch_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        return dict(results)
    def ask_knowledge_graph_query(queries: List[str], project_id: str, node_ids: List[str] = []) -> Dict[str, str]:
        """
        Query the code knowledge graph using multiple natural language questions.
        The knowledge graph contains information about every function, class, and file in the codebase.
        This method allows asking multiple questions about the codebase in a single operation.

        Inputs:
        - queries (List[str]): A list of natural language questions that the user wants to ask the knowledge graph.
          Each question should be clear and concise, related to the codebase.
        - project_id (str): The ID of the project being evaluated, this is a UUID.

        Returns:
        - Dict[str, str]: A dictionary where keys are the original queries and values are the corresponding responses.
        """
        query_list = [QueryRequest(query=query, project_id=project_id, node_ids=node_ids) for query in queries]
        return asyncio.run(CodeTools.ask_multiple_knowledge_graph_queries(query_list))

    @classmethod
    def get_kg_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            StructuredTool.from_function(
                func=cls.get_nodes_from_tags,
                name="Get Nodes from Tags",
                description="""
                Fetch nodes from the knowledge graph based on specified tags. Use this tool to retrieve nodes of specific types for a project.

                Input:
                - tags (List[str]): A list of tags to filter nodes. Valid tags are:
                API, WEBSOCKET, PRODUCER, CONSUMER, DATABASE, SCHEMA, HTTP
                - project_id (str): The UUID of the project being evaluated

                Usage guidelines:
                1. Use for broad queries requiring ALL nodes of specific types.
                2. Limit to 1-2 tags per query for best results.
                3. Returns file paths, docstrings, text, node IDs, and names.
                4. List cannot be empty.

                Example: To find all API endpoints, use tags=['API']""",
                args_schema=GetNodesFromTagsInput,
            ),
            StructuredTool.from_function(
                func=cls.ask_knowledge_graph_query,
                name="Ask Knowledge Graph Queries",
                description="""
            Query the code knowledge graph using multiple natural language questions.
            The knowledge graph contains information about every function, class, and file in the codebase.
            This tool allows asking multiple questions about the codebase in a single operation.
            
            Inputs:
            - queries (List[str]): A list of natural language questions to ask the knowledge graph. Each question should be 
            clear and concise, related to the codebase, such as "What does the XYZ class do?" or "How is the ABC function used?"
            - project_id (str): The ID of the project being evaluated, this is a UUID.
            - node_ids (List[str]): A list of node ids to query, this is an optional parameter that can be used to query a specific node. use this only when you are sure that the answer to the question is related to that node.
            
            Use this tool when you need to ask multiple related questions about the codebase at once.
            Do not use this to query code directly.""",
                args_schema=MultipleKnowledgeGraphQueriesInput,
            ),
        ]
