import asyncio
import logging
from typing import Any, Dict

from fastapi import HTTPException
from langchain_core.tools import StructuredTool, Tool
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class GetCodeFromNodeNameTool:
    name = "get_code_from_node_name"
    description = """Retrieves code for a specific node in a repository given its node name.
        :param project_id: string, the repository ID (UUID).
        :param node_name: string, the name of the node to retrieve code from.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_name": "src/services/UserService.ts:authenticateUser"
            }

        Returns dictionary containing node details including code content and file location.
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

    async def arun(self, project_id: str, node_name: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_name)

    def run(self, project_id: str, node_name: str) -> Dict[str, Any]:
        project = asyncio.run(
            ProjectService(self.sql_db).get_project_repo_details_from_db(
                project_id, self.user_id
            )
        )

        if not project:
            raise ValueError(
                f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
            )

        try:
            node_data = self.get_node_data(project_id, node_name)
            if not node_data:
                logger.error(
                    f"Node with name '{node_name}' not found in repo '{project_id}'"
                )
                return {
                    "error": f"Node with name '{node_name}' not found in repo '{project_id}'"
                }

            project = self._get_project(project_id)
            if not project:
                logger.error(f"Project with ID '{project_id}' not found in database")
                return {
                    "error": f"Project with ID '{project_id}' not found in database"
                }

            return self._process_result(node_data, project, node_name)
        except Exception as e:
            logger.error(
                f"Project: {project_id} Unexpected error in GetCodeFromNodeNameTool: {str(e)}"
            )
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def get_node_data(self, project_id: str, node_name: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE toLower(n.name) = toLower($node_name)
        RETURN n.file_path AS file, n.start_line AS start_line, n.end_line AS end_line, n.node_id AS node_id
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_name=node_name, project_id=project_id)
            return result.single()

    def _get_project(self, project_id: str) -> Project:
        return self.sql_db.query(Project).filter(Project.id == project_id).first()

    def _process_result(
        self, node_data: Dict[str, Any], project: Project, node_name: str
    ) -> Dict[str, Any]:
        file_path = node_data["file"]
        start_line = node_data["start_line"]
        end_line = node_data["end_line"]

        relative_file_path = self._get_relative_file_path(file_path)

        code_provider_service = CodeProviderService(self.sql_db)
        try:
            code_content = code_provider_service.get_file_content(
                project.repo_name,
                relative_file_path,
                start_line,
                end_line,
                project.branch_name,
                project.id,
            )
        except HTTPException as http_exc:
            return {"error": f"Failed to retrieve code content: {http_exc.detail}"}

        return {
            "repo_name": project.repo_name,
            "branch_name": project.branch_name,
            "node_name": node_name,
            "file_path": file_path,
            "relative_file_path": relative_file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_content": code_content,
        }

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        parts = file_path.split("/")
        try:
            projects_index = parts.index("projects")
            return "/".join(parts[projects_index + 2 :])
        except ValueError:
            logger.warning(f"'projects' not found in file path: {file_path}")
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_code_from_node_name_tool(sql_db: Session, user_id: str) -> Tool:
    tool_instance = GetCodeFromNodeNameTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Code From Node Name",
        description="Retrieves code for a specific node in a repository given its node name",
    )
