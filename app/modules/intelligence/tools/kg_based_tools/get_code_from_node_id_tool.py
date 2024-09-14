from typing import Any, Dict, List

from langchain.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.github.github_service import GithubService
from app.modules.projects.projects_model import Project


class GetCodeFromNodeIdInput(BaseModel):
    repo_id: str = Field(description="The repository ID")
    node_id: str = Field(description="The node ID")


class GetCodeFromMultipleNodeIdsInput(BaseModel):
    repo_id: str = Field(description="The repository ID")
    node_ids: List[str] = Field(description="List of node IDs")


class GetCodeFromNodeIdTool:
    name = "get_code_from_node_id"
    description = (
        "Retrieves code for a specific node id in a repository given its node ID"
    )

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    def run(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        try:
            node_data = self._get_node_data(repo_id, node_id)
            if not node_data:
                print(f"Node with ID '{node_id}' not found in repo '{repo_id}'")
                return {
                    "error": f"Node with ID '{node_id}' not found in repo '{repo_id}'"
                }

            project = self._get_project(repo_id)
            if not project:
                print(f"Project with ID '{repo_id}' not found in database")
                return {"error": f"Project with ID '{repo_id}' not found in database"}

            return self._process_result(node_data, project, node_id)
        except Exception as e:
            print(f"Unexpected error in GetCodeFromNodeIdTool: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def run_multiple(self, repo_id: str, node_ids: List[str]) -> Dict[str, Any]:
        try:
            project = self._get_project(repo_id)
            if not project:
                print(f"Project with ID '{repo_id}' not found in database")
                return {"error": f"Project with ID '{repo_id}' not found in database"}

            results = {}
            for node_id in node_ids:
                node_data = self._get_node_data(repo_id, node_id)
                if node_data:
                    results[node_id] = self._process_result(node_data, project, node_id)
                else:
                    results[node_id] = {
                        "error": f"Node with ID '{node_id}' not found in repo '{repo_id}'"
                    }

            return results
        except Exception as e:
            print(f"Unexpected error in GetCodeFromNodeIdTool (multiple): {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

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
        if "code" in node_data and node_data["code"] is None:
            code_content = GithubService.get_file_content(
                project.repo_name,
                relative_file_path,
                start_line,
                end_line,
            )
        else:
            code_content = node_data["code"]

        if "docstring" in node_data and node_data["docstring"] is None:
            docstring = GithubService.get_file_content(
                project.repo_name,
                relative_file_path,
                start_line,
                end_line,
            )
        else:
            docstring = node_data["docstring"]

        return {
            "repo_name": project.repo_name,
            "branch_name": project.branch_name,
            "node_id": node_id,
            "file_path": file_path,
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
            print(f"'projects' not found in file path: {file_path}")
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()

    async def arun(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        return self.run(repo_id, node_id)

    async def arun_multiple(self, repo_id: str, node_ids: List[str]) -> Dict[str, Any]:
        return self.run_multiple(repo_id, node_ids)


def get_tool(sql_db: Session) -> List[StructuredTool]:
    """
    Get StructuredTool objects for the GetCodeFromNodeIdTool.
    """
    tool_instance = GetCodeFromNodeIdTool(sql_db)
    return [
        StructuredTool.from_function(
            func=tool_instance.run,
            name="Get Code and docstring From Node ID",
            description="Retrieves code for a specific node id in a repository given its node ID",
            args_schema=GetCodeFromNodeIdInput,
        ),
        StructuredTool.from_function(
            func=tool_instance.run_multiple,
            name="Get Code and docstring From Multiple Node IDs",
            description="Retrieves code for multiple node ids in a repository given their node IDs",
            args_schema=GetCodeFromMultipleNodeIdsInput,
        ),
    ]
