import os
from typing import List

import requests
from langchain.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from app.core.config_provider import ConfigProvider
from app.core.database import get_db
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService


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


class CodeTools:
    @staticmethod
    def ask_knowledge_graph(query: str, project_id: str) -> str:
        """
        Query the code knowledge graph using natural language questions.
        The knowledge graph contains information about every function, class, and file in the codebase.
        Ask it questions about the codebase.
        Inputs for the ask_knowledge_graph method:
        - query (str): A natural language question that the user wants to ask the knowledge graph. This should be a clear and concise question related to the codebase, such as "What functions are defined in the project?" or "How do I use the XYZ class?".

        - project_id (str): The ID of the project being evaluated, this is a UUID.
        """
        data = {"project_id": project_id, "query": query}
        headers = {"Content-Type": "application/json"}
        kg_query_url = os.getenv("KNOWLEDGE_GRAPH_URL")
        response = requests.post(kg_query_url, json=data, headers=headers)
        return response.json()

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

    @classmethod
    def get_kg_tools(cls) -> List[Tool]:
        """
        Get a list of LangChain Tool objects for use in agents.
        """
        return [
            StructuredTool.from_function(
                func=cls.ask_knowledge_graph,
                name="Ask Knowledge Graph",
                description=""""
        Query the code knowledge graph using natural language questions.
        The knowledge graph contains information about every function, class, and file in the codebase.
        Ask it questions about the codebase.
        Inputs for the ask_knowledge_graph method:
        - query (str): A natural language question that the user wants to ask the knowledge graph. This should be a clear and concise question related to the codebase, such as "What functions are defined in the project?" or "How do I use the XYZ class?".
        - project_id (str): The ID of the project being evaluated, this is a UUID.
        Do not use this to query code directly.""",
                args_schema=KnowledgeGraphInput,
            ),
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

                Example: To find all API endpoints, use tags=['API']""",
                args_schema=GetNodesFromTagsInput,
            ),
        ]
