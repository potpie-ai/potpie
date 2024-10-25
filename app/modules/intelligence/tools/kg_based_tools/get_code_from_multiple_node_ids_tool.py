import asyncio
import logging
from typing import Any, Dict, List

from langchain.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.github.github_service import GithubService
from app.modules.projects.projects_model import Project

logger = logging.getLogger(__name__)


class GetCodeFromMultipleNodeIdsInput(BaseModel):
    repo_id: str = Field(description="The repository ID, this is a UUID")
    node_ids: List[str] = Field(description="List of node IDs, this is a UUID")


class GetCodeFromMultipleNodeIdsTool:
    name = "get_code_from_multiple_node_ids"
    description = (
        "Retrieves code for multiple node ids in a repository given their node IDs"
    )

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

    def run(self, repo_id: str, node_ids: List[str]) -> Dict[str, Any]:
        return asyncio.run(self.run_multiple(repo_id, node_ids))

    async def run_multiple(self, repo_id: str, node_ids: List[str]) -> Dict[str, Any]:
        try:
            project = self._get_project(repo_id)
            if not project:
                logger.error(f"Project with ID '{repo_id}' not found in database")
                return {"error": f"Project with ID '{repo_id}' not found in database"}
            if project.user_id != self.user_id:
                raise ValueError(
                    f"Project with ID '{repo_id}' not found in database for user '{self.user_id}'"
                )

            tasks = [
                self._retrieve_node_data(repo_id, node_id, project)
                for node_id in node_ids
            ]
            completed_tasks = await asyncio.gather(*tasks)

            return {
                node_id: result for node_id, result in zip(node_ids, completed_tasks)
            }
        except Exception as e:
            logger.error(
                f"Unexpected error in GetCodeFromMultipleNodeIdsTool: {str(e)}"
            )
            return {"error": f"An unexpected error occurred: {str(e)}"}

    async def _retrieve_node_data(
        self, repo_id: str, node_id: str, project: Project
    ) -> Dict[str, Any]:
        node_data = self._get_node_data(repo_id, node_id)
        if node_data:
            return self._process_result(node_data, project, node_id)
        else:
            return {"error": f"Node with ID '{node_id}' not found in repo '{repo_id}'"}

    def _get_node_data(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (n:NODE {node_id: $node_id, repoId: $repo_id})
        RETURN n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.text as code, n.docstring as docstring
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, repo_id=repo_id)
            return result.single()

    def _get_project(self, repo_id: str) -> Project:
        return self.sql_db.query(Project).filter(Project.id == repo_id).first()

    def _process_result(
        self, node_data: Dict[str, Any], project: Project, node_id: str
    ) -> Dict[str, Any]:
        file_path = node_data["file_path"]
        start_line = node_data["start_line"]
        end_line = node_data["end_line"]

        relative_file_path = self._get_relative_file_path(file_path)

        code_content = GithubService(self.sql_db).get_file_content(
            project.repo_name,
            relative_file_path,
            start_line,
            end_line,
            project.branch_name,
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
            logger.warning(f"'projects' not found in file path: {file_path}")
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_code_from_multiple_node_ids_tool(
    sql_db: Session, user_id: str
) -> StructuredTool:
    tool_instance = GetCodeFromMultipleNodeIdsTool(sql_db, user_id)
    return StructuredTool.from_function(
        func=tool_instance.run,
        name="Get Code and docstring From Multiple Node IDs",
        description="""Retrieves code and docstring for multiple node ids in a repository given their node IDs
                Inputs for the run_multiple method:
                - repo_id (str): The repository ID to retrieve code and docstring for, this is a UUID.
                - node_ids (List[str]): A list of node IDs to retrieve code and docstring for, this is a UUID.""",
        args_schema=GetCodeFromMultipleNodeIdsInput,
    )
