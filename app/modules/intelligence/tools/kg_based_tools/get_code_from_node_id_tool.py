import asyncio
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_model import Project
from app.modules.intelligence.tools.tool_utils import truncate_dict_response
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GetCodeFromNodeIdInput(BaseModel):
    project_id: str = Field(description="The repository ID, this is a UUID")
    node_id: str = Field(description="The node ID, this is a UUID")


class GetCodeFromNodeIdTool:
    name = "Get Code and docstring From Node ID"
    description = """Retrieves code and docstring for a specific node in a repository.
        :param project_id: string, the repository ID (UUID).
        :param node_id: string, the node ID to retrieve code for.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000"
            }

        Returns dictionary containing node code, docstring, and file location details.
        """

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        uri = neo4j_config.get("uri")
        username = neo4j_config.get("username")
        password = neo4j_config.get("password")
        if not uri or not username or not password:
            raise ValueError("Neo4j configuration is incomplete")
        return GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

    async def arun(self, project_id: str, node_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_id)

    def run(self, project_id: str, node_id: str) -> Dict[str, Any]:
        """Synchronous version that handles the core logic"""
        try:
            node_data = self._get_node_data(project_id, node_id)
            if not node_data:
                logger.warning(
                    f"Node with ID '{node_id}' not found in repo '{project_id}'"
                )
                return {
                    "error": f"Node with ID '{node_id}' not found in repo '{project_id}'"
                }

            project = self._get_project(project_id)
            if not project:
                logger.error(f"Project with ID '{project_id}' not found in database")
                return {
                    "error": f"Project with ID '{project_id}' not found in database"
                }
            if project.user_id != self.user_id:
                raise ValueError(
                    f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
                )

            return self._process_result(node_data, project, node_id)
        except Exception:
            logger.exception(
                "Unexpected error in GetCodeFromNodeIdTool",
                project_id=project_id,
                node_id=node_id,
                user_id=self.user_id,
            )
            return {"error": "An unexpected error occurred"}

    def _get_node_data(self, project_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
        RETURN n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.text as code, n.docstring as docstring
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, project_id=project_id)
            return result.single()

    def _get_project(self, project_id: str) -> Project:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID '{project_id}' not found")
        return project

    def _process_result(
        self, node_data: Dict[str, Any], project: Project, node_id: str
    ) -> Dict[str, Any]:
        # Check if node_data has the required fields
        if not node_data or "file_path" not in node_data:
            logger.error(
                f"Node data is incomplete or missing file_path for node_id: {node_id}"
            )
            return {"error": f"Node data is incomplete for node_id: {node_id}"}

        file_path = node_data["file_path"]
        if file_path is None:
            logger.error(f"File path is None for node_id: {node_id}")
            return {"error": f"File path is None for node_id: {node_id}"}

        start_line = node_data["start_line"]
        end_line = node_data["end_line"]

        relative_file_path = self._get_relative_file_path(file_path)

        code_content = CodeProviderService(self.sql_db).get_file_content(
            project.repo_name,
            relative_file_path,
            start_line,
            end_line,
            project.branch_name,
            project.id,
            project.commit_id,
        )

        docstring = None
        if node_data.get("docstring", None):
            docstring = node_data["docstring"]

        result = {
            "node_id": node_id,
            "file_path": relative_file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_content": code_content,
            "docstring": docstring,
        }

        # Truncate response if it exceeds character limits
        truncated_result = truncate_dict_response(result)
        if len(str(result)) > 80000:
            logger.warning(
                f"get_code_from_node_id output truncated for node_id={node_id}, project_id={project.id}"
            )
        return truncated_result

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        parts = file_path.split("/")
        try:
            projects_index = parts.index("projects")
            return "/".join(parts[projects_index + 2 :])
        except ValueError:
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_code_from_node_id_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = GetCodeFromNodeIdTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Code and docstring From Node ID",
        description="""Retrieves code and docstring for a specific node id in a repository given its node ID
                       Inputs for the run method:
                       - project_id (str): The repository ID to retrieve code and docstring for, this is a UUID.
                       - node_id (str): The node ID to retrieve code and docstring for, this is a UUID.

                       ⚠️ IMPORTANT: Large code content may result in truncated responses (max 80,000 characters).
                       If the response is truncated, a notice will be included indicating the truncation occurred.""",
        args_schema=GetCodeFromNodeIdInput,
    )
