import asyncio
from typing import List

from langchain_core.tools import StructuredTool
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
    name = "Get Nodes from Tags"
    description = """Fetch nodes from the knowledge graph based on specified tags.
        :param tags: array, list of tags to filter nodes by. Valid tags are: API, WEBSOCKET, PRODUCER, CONSUMER, DATABASE, SCHEMA, EXTERNAL_SERVICE, CONFIGURATION, SCRIPT.
        :param project_id: string, the project ID (UUID).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "tags": ["API", "DATABASE"]
            }

        Returns list of nodes with:
        - file_path: string - path to the file
        - docstring: string - documentation if available
        - text: string - node text content
        - node_id: string - unique identifier
        - name: string - node name

        Usage guidelines:
        1. Use for broad queries requiring ALL nodes of specific types
        2. Limit to 1-2 tags per query for best results
        3. List cannot be empty
        """

    def __init__(self, sql_db, user_id):
        self.sql_db = sql_db
        self.user_id = user_id

    async def arun(self, tags: List[str], project_id: str) -> str:
        return await asyncio.to_thread(self.run, tags, project_id)

    def run(self, tags: List[str], project_id: str) -> str:
        """
        Get nodes from the knowledge graph based on the provided tags.
        Inputs for the fetch_nodes method:
        - tags (List[str]): A list of tags to filter the nodes by.
           Backend Tags:
           * API: Does the code define any API endpoint? Look for route definitions, request handling.
           * AUTH: Does the code handle authentication or authorization? Check for auth middleware, token validation.
           * DATABASE: Does the code interact with a database? Look for query execution, data operations.
           * UTILITY: Does the code provide helper or utility functions? Check for reusable functions.
           * PRODUCER: Does the code generate and send messages to a queue/topic? Look for message publishing.
           * CONSUMER: Does the code receive and process messages from a queue/topic? Check for message handling.
           * EXTERNAL_SERVICE: Does the code integrate with external services? Check for HTTP client usage.
           * CONFIGURATION: Does the code manage configuration settings? Look for config management.

           Frontend Tags:
           * UI_COMPONENT: Does the code render visual components? Check for UI rendering logic.
           * FORM_HANDLING: Does the code manage forms? Look for form submission and validation.
           * STATE_MANAGEMENT: Does the code manage app/component state? Check for state logic.
           * DATA_BINDING: Does the code bind data to UI elements? Look for data binding patterns.
           * ROUTING: Does the code handle frontend navigation? Check for routing logic.
           * EVENT_HANDLING: Does the code handle user interactions? Look for event handlers.
           * STYLING: Does the code apply styling/theming? Check for style-related code.
           * MEDIA: Does the code manage media assets? Look for image/video handling.
           * ANIMATION: Does the code define UI animations? Check for animation logic.
           * ACCESSIBILITY: Does the code implement a11y features? Look for accessibility code.
           * DATA_FETCHING: Does the code fetch frontend data? Check for data retrieval logic.
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
        coroutine=GetNodesFromTags(sql_db, user_id).arun,
        func=GetNodesFromTags(sql_db, user_id).run,
        name="Get Nodes from Tags",
        description="""
        Fetch nodes from the knowledge graph based on specified tags. Use this tool to retrieve nodes of specific types for a project.

        Input:
        - tags (List[str]): A list of tags to filter nodes. Valid tags include:
        API, AUTH, DATABASE, UTILITY, PRODUCER, CONSUMER, EXTERNAL_SERVICE, CONFIGURATION
        UI_COMPONENT, FORM_HANDLING, STATE_MANAGEMENT, DATA_BINDING, ROUTING,
        EVENT_HANDLING, STYLING, MEDIA, ANIMATION, ACCESSIBILITY, DATA_FETCHING

        - project_id (str): The UUID of the project being evaluated

        Usage guidelines:
        1. Use for broad queries requiring ALL nodes of specific types.
        2. Limit to 1-2 tags per query for best results.
        3. Returns file paths, docstrings, text, node IDs, and names.
        4. List cannot be empty.

        Example: To find all API endpoints, use tags=['API']""",
        args_schema=GetNodesFromTagsInput,
    )
