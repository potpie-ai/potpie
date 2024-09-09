from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.github.github_service import GithubService
from app.core.config_provider import config_provider
from sqlalchemy.orm import Session

class GetCodeFromNodeNameInput(BaseModel):
    repo_name: str = Field(..., description="The name of the repository")
    node_name: str = Field(..., description="The name of the node")

class GetCodeFromNodeNameTool(BaseTool):
    name = "get_code_from_node_name"
    description = "Retrieves code for a specific node in a repository given its node name"
    args_schema = GetCodeFromNodeNameInput
    
    _code_graph_service: CodeGraphService = PrivateAttr()

    def __init__(self, sql_db: Session):
        super().__init__()
        neo4j_config = config_provider.get_neo4j_config()
        self._code_graph_service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    sql_db,
        )

    def _run(self, repo_name: str, node_name: str) -> dict:
        query = (
            f"MATCH (n:NODE {{name: '{node_name}', repoId: '{repo_name}'}}) "
            "RETURN n.file AS file, n.start_line AS start_line, n.end_line AS end_line"
        )
        result = self._code_graph_service.query_graph(query)

        if not result:
            return {"error": f"Node '{node_name}' not found in repo '{repo_name}'"}

        node = result[0]
        file_path = node['file']
        start_line = node['start_line']
        end_line = node['end_line']

        code_content = GithubService.get_file_content(repo_name, file_path, start_line, end_line)
        
        return {
            "repo_name": repo_name,
            "node_name": node_name,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "code_content": code_content
        }

    async def _arun(self, repo_name: str, node_name: str) -> dict:
        # This tool doesn't support async operations
        return self._run(repo_name, node_name)
