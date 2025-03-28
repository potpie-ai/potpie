import asyncio
import logging
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)


class GetCodeFromProbableNodeNameInput(BaseModel):
    project_id: str = Field(description="The project ID, this is a UUID")
    probable_node_names: List[str] = Field(
        description="List of probable node names in the format of 'file_path:function_name' or 'file_path:class_name' or 'file_path'"
    )


class GetCodeFromProbableNodeNameTool:
    name = "Get Code and docstring From Probable Node Name"
    description = """Retrieves code for nodes matching probable names in a repository.
        :param project_id: string, the project ID (UUID).
        :param probable_node_names: array, list of probable node names in format 'file_path:function_name' or 'file_path:class_name'.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "probable_node_names": [
                    "src/services/auth.ts:validateToken",
                    "src/models/User.ts:User"
                ]
            }

        Returns list of matching nodes with their code content and metadata.
        """

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.neo4j_driver = self._create_neo4j_driver()
        self.search_service = SearchService(self.sql_db)

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    async def process_probable_node_name(
        self, project_id: str, probable_node_name: str
    ):
        try:
            node_id_query = " ".join(
                probable_node_name.replace("/", " ").replace(":", " ").split()
            )
            relevance_search = await self.search_service.search_codebase(
                project_id, node_id_query
            )
            node_id = None
            if relevance_search:
                node_id = relevance_search[0]["node_id"]

            if not node_id:
                return {
                    "error": f"Node with name '{probable_node_name}' not found in project '{project_id}'"
                }

            return await self.execute(project_id, node_id)
        except Exception as e:
            logger.error(
                f"Unexpected error in GetCodeFromProbableNodeNameTool: {str(e)}"
            )
            return {"error": f"An unexpected error occurred: {str(e)}"}

    async def find_node_from_probable_name(
        self, project_id: str, probable_node_names: List[str]
    ) -> List[Dict[str, Any]]:
        tasks = [
            self.process_probable_node_name(project_id, name)
            for name in probable_node_names
        ]
        return await asyncio.gather(*tasks)

    async def arun(
        self, project_id: str, probable_node_names: List[str]
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.run, project_id, probable_node_names)

    def run(
        self, project_id: str, probable_node_names: List[str]
    ) -> List[Dict[str, Any]]:
        return asyncio.run(
            asyncio.to_thread(
                self.get_code_from_probable_node_name, project_id, probable_node_names
            )
        )

    def get_code_from_probable_node_name(
        self, project_id: str, probable_node_names: List[str]
    ) -> List[Dict[str, Any]]:
        project = asyncio.run(
            ProjectService(self.sql_db).get_project_repo_details_from_db(
                project_id, self.user_id
            )
        )
        if not project:
            raise ValueError(
                f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
            )
        return asyncio.run(
            self.find_node_from_probable_name(project_id, probable_node_names)
        )

    async def execute(self, project_id: str, node_id: str) -> Dict[str, Any]:
        return self.internal_run(project_id, node_id)

    def internal_run(self, project_id: str, node_id: str) -> Dict[str, Any]:
        try:
            node_data = self._get_node_data(project_id, node_id)
            if not node_data:
                logger.error(
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

            return self._process_result(node_data, project, node_id)
        except Exception as e:
            logger.warn(
                f"Unexpected error in GetCodeFromProbableNodeNameTool: {str(e)}"
            )
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_node_data(self, project_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
        RETURN n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.text as code, n.docstring as docstring
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, project_id=project_id)
            return result.single()

    def _get_project(self, project_id: str) -> Project:
        return self.sql_db.query(Project).filter(Project.id == project_id).first()

    def _process_result(
        self, node_data: Dict[str, Any], project: Project, node_id: str
    ) -> Dict[str, Any]:
        file_path = node_data["file_path"]
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
        )

        docstring = None
        if node_data.get("docstring", None):
            docstring = node_data["docstring"]

        return {
            "node_id": node_id,
            "relative_file_path": relative_file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_content": code_content,
            "docstring": docstring,
        }

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


def get_code_from_probable_node_name_tool(
    sql_db: Session, user_id: str
) -> StructuredTool:
    tool_instance = GetCodeFromProbableNodeNameTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Code and docstring From Probable Node Name",
        description="""Retrieves code for nodes matching probable names in a repository.
        :param project_id: string, the project ID (UUID).
        :param probable_node_names: array, list of probable node names in format 'file_path:function_name' or 'file_path:class_name'.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "probable_node_names": [
                    "src/services/auth.ts:validateToken",
                    "src/models/User.ts:User"
                ]
            }

        Returns list of matching nodes with their code content and metadata.
        """,
        args_schema=GetCodeFromProbableNodeNameInput,
    )
