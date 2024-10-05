import asyncio
from typing import List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.config_provider import ConfigProvider
from app.core.database import get_db
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.projects.projects_service import ProjectService


class GetNodesFromTagsInput(BaseModel):
    tags: List[str] = Field(description="A list of tags to filter the nodes by")
    project_id: str = Field(
        description="The project id metadata for the project being evaluated"
    )


class GetNodesFromTags:
    def __init__(self, sql_db, user_id):
        self.sql_db = sql_db
        self.user_id = user_id

    def fetch_nodes(self, tags: List[str], project_id: str) -> str:
        """
        Get nodes from the knowledge graph based on the provided tags.
        Inputs for the fetch_nodes method:
        - tags (List[str]): A list of tags to filter the nodes by.
           * API: Does the code define any API endpoint? Look for route definitions, request handling, or API client usage.
           * WEBSOCKET: Does the code implement or use WebSocket connections? Check for WebSocket-specific libraries or protocols.
           * PRODUCER: Does the code generate and send messages to a queue or topic? Look for message publishing or event emission.
           * CONSUMER: Does the code receive and process messages from a queue or topic? Check for message subscription or event handling.
           * DATABASE: Does the code interact with a database? Look for query execution, data insertion, updates, or deletions.
           * SCHEMA: Does the code define any database schema? Look for ORM models, table definitions, or schema-related code.
           * EXTERNAL_SERVICE: Does the code make HTTP requests to external services? Check for HTTP client usage or request handling.
        - project_id (str): The ID of the project being evaluated, this is a UUID.
        """
        project = asyncio.run(
            ProjectService(self.sql_db).get_project_repo_details_from_db(
                project_id, self.user_id
            )
        )
        if not project:
            raise ValueError(
                f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
            )
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


def get_nodes_from_tags_tool(sql_db, user_id) -> StructuredTool:
    return StructuredTool.from_function(
        func=GetNodesFromTags(sql_db, user_id).fetch_nodes,
        name="Get Nodes from Tags",
        description="""
        Fetch nodes from the knowledge graph based on specified tags. Use this tool to retrieve nodes of specific types for a project.

        Input:
        - tags (List[str]): A list of tags to filter nodes. Valid tags are:
        API, WEBSOCKET, PRODUCER, CONSUMER, DATABASE, SCHEMA, EXTERNAL_SERVICE, CONFIGURATION, SCRIPT
        - project_id (str): The UUID of the project being evaluated

        Usage guidelines:
        1. Use for broad queries requiring ALL nodes of specific types.
        2. Limit to 1-2 tags per query for best results.
        3. Returns file paths, docstrings, text, node IDs, and names.
        4. List cannot be empty.

        Example: To find all API endpoints, use tags=['API']""",
        args_schema=GetNodesFromTagsInput,
    )
