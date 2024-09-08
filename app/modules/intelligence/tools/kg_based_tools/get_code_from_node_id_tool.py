from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.github.github_service import GithubService

class GetCodeFromNodeIdInput(BaseModel):
    repo_name: str = Field(..., description="The name of the repository")
    node_id: str = Field(..., description="The ID of the node")

class GetCodeFromNodeIdTool(BaseTool):
    name = "get_code_from_node_id"
    description = "Retrieves code for a specific node in a repository given its node ID"
    args_schema = GetCodeFromNodeIdInput

    def __init__(self, db: Session):
        super().__init__()
        self.db = db
        self.code_graph_service = CodeGraphService(db)

    def _run(self, repo_name: str, node_id: str) -> dict:
        query = (
            f"MATCH (n:NODE {{node_id: '{node_id}', repoId: '{repo_name}'}}) "
            "RETURN n.file AS file, n.start_line AS start_line, n.end_line AS end_line"
        )
        result = self.code_graph_service.query_graph(query)

        if not result:
            return {"error": f"Node with ID '{node_id}' not found in repo '{repo_name}'"}

        node = result[0]
        file_path = node['file']
        start_line = node['start_line']
        end_line = node['end_line']

        code_content = GithubService.get_file_content(repo_name, file_path, start_line, end_line)
        
        return {
            "repo_name": repo_name,
            "node_id": node_id,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_content": code_content
        }

    async def _arun(self, repo_name: str, node_id: str) -> dict:
        # This tool doesn't support async operations
        return self._run(repo_name, node_id)
