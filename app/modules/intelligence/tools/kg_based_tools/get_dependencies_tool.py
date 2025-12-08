import asyncio
import logging
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class GetDependsOnInput(BaseModel):
    project_id: str = Field(description="The project ID, this is a UUID")
    node_id: str = Field(
        description="The node ID to find dependencies for, this is a UUID"
    )


class GetDependenciesTool:
    name = "Get Dependencies"
    description = """Retrieves nodes that a given node depends on (outgoing dependencies).
        This includes:
        - Functions/methods that this node calls (CALLS relationship)
        - Classes that this node inherits from (REFERENCES to class definitions)
        - Libraries/modules that this node imports (REFERENCES)

        :param project_id: string, the project ID (UUID).
        :param node_id: string, the node ID to find dependencies for.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000"
            }

        Returns list of nodes that the given node depends on with their details.
        """

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    async def arun(self, project_id: str, node_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_id)

    def run(self, project_id: str, node_id: str) -> Dict[str, Any]:
        try:
            project = asyncio.run(
                ProjectService(self.sql_db).get_project_repo_details_from_db(
                    project_id, self.user_id
                )
            )
            if not project:
                raise ValueError(
                    f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
                )

            dependencies = self._get_dependencies(project_id, node_id)

            if not dependencies:
                return {
                    "node_id": node_id,
                    "project_id": project_id,
                    "depends_on": [],
                    "message": "No outgoing dependencies found for this node",
                }

            return {
                "node_id": node_id,
                "project_id": project_id,
                "depends_on": dependencies,
                "count": len(dependencies),
            }
        except Exception as e:
            logger.error(f"Error in GetDependsOnTool: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}

    def _get_dependencies(self, project_id: str, node_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (source:NODE {node_id: $node_id, repoId: $project_id})
        OPTIONAL MATCH (source)-[r:CALLS|REFERENCES]->(target:NODE {repoId: $project_id})
        WHERE target IS NOT NULL AND source <> target
        RETURN DISTINCT
            target.node_id AS node_id,
            target.name AS name,
            target.file_path AS file_path,
            target.start_line AS start_line,
            target.end_line AS end_line,
            target.docstring AS docstring,
            type(r) AS relationship_type,
            labels(target) AS labels
        ORDER BY relationship_type, name
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, project_id=project_id)
            dependencies = []
            for record in result:
                if record["node_id"]:
                    dependencies.append(
                        {
                            "node_id": record["node_id"],
                            "name": record["name"],
                            "file_path": record["file_path"],
                            "start_line": record["start_line"],
                            "end_line": record["end_line"],
                            "docstring": record["docstring"],
                            "relationship_type": record["relationship_type"],
                            "node_type": record["labels"][0]
                            if record["labels"]
                            else None,
                        }
                    )
            return dependencies

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_dependencies_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = GetDependenciesTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Dependencies",
        description="""Retrieves nodes that a given node depends on (outgoing dependencies).
                       This includes functions/methods called, classes inherited from, and libraries imported.

                       Inputs:
                       - project_id (str): The project ID (UUID)
                       - node_id (str): The node ID to find dependencies for (UUID)

                       Returns list of dependent nodes with their file paths, docstrings, and relationship types.""",
        args_schema=GetDependsOnInput,
    )
