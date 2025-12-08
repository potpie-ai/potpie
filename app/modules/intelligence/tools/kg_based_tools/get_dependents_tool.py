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


class GetDependedByInput(BaseModel):
    project_id: str = Field(description="The project ID, this is a UUID")
    node_id: str = Field(description="The node ID to find dependents for, this is a UUID")


class GetDependentsTool:
    name = "Get Dependents"
    description = """Retrieves nodes that depend on a given node (incoming dependencies).
        This includes:
        - Functions/methods that call this node (CALLS relationship)
        - Classes that inherit from this node (REFERENCES from child classes)
        - Files/modules that import this node (REFERENCES)
        
        :param project_id: string, the project ID (UUID).
        :param node_id: string, the node ID to find dependents for.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000"
            }

        Returns list of nodes that depend on the given node with their details.
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

            dependents = self._get_dependents(project_id, node_id)
            
            if not dependents:
                return {
                    "node_id": node_id,
                    "project_id": project_id,
                    "depended_by": [],
                    "message": "No incoming dependencies found for this node"
                }

            return {
                "node_id": node_id,
                "project_id": project_id,
                "depended_by": dependents,
                "count": len(dependents)
            }
        except Exception as e:
            logger.error(f"Error in GetDependedByTool: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}

    def _get_dependents(self, project_id: str, node_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (target:NODE {node_id: $node_id, repoId: $project_id})
        OPTIONAL MATCH (source:NODE {repoId: $project_id})-[r:CALLS|REFERENCES]->(target)
        WHERE source IS NOT NULL AND source <> target
        RETURN DISTINCT 
            source.node_id AS node_id,
            source.name AS name,
            source.file_path AS file_path,
            source.start_line AS start_line,
            source.end_line AS end_line,
            source.docstring AS docstring,
            type(r) AS relationship_type,
            labels(source) AS labels
        ORDER BY relationship_type, name
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, project_id=project_id)
            dependents = []
            for record in result:
                if record["node_id"]:
                    dependents.append({
                        "node_id": record["node_id"],
                        "name": record["name"],
                        "file_path": record["file_path"],
                        "start_line": record["start_line"],
                        "end_line": record["end_line"],
                        "docstring": record["docstring"],
                        "relationship_type": record["relationship_type"],
                        "node_type": record["labels"][0] if record["labels"] else None
                    })
            return dependents

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_dependents_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = GetDependentsTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Dependents",
        description="""Retrieves nodes that depend on a given node (incoming dependencies).
                       This includes functions/methods that call this node, classes that inherit from it, 
                       and files/modules that import it.
                       
                       Inputs:
                       - project_id (str): The project ID (UUID)
                       - node_id (str): The node ID to find dependents for (UUID)
                       
                       Returns list of nodes that depend on this node with their file paths, docstrings, and relationship types.""",
        args_schema=GetDependedByInput,
    )
