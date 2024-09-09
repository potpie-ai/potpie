from typing import Dict, Any, List
from sqlalchemy.orm import Session
from neo4j import GraphDatabase
from app.modules.github.github_service import GithubService
from app.core.config_provider import config_provider

class GetCodeFromNodeNameTool:
    name = "get_code_from_node_name"
    description = "Retrieves code for a specific node in a repository given its node name"

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db

    def run(self, repo_name: str, node_name: str) -> Dict[str, Any]:
        query = (
            "MATCH (n:NODE {name: $node_name, repoId: $repo_name}) "
            "RETURN n.file AS file, n.start_line AS start_line, n.end_line AS end_line"
        )
        result = self.query_graph(query, node_name=node_name, repo_name=repo_name)
        return self.process_query_result(result, repo_name, node_name)

    async def arun(self, repo_name: str, node_name: str) -> Dict[str, Any]:
        return self.run(repo_name, node_name)

    @staticmethod
    def query_graph(query: str, **params) -> List[Dict[str, Any]]:
        neo4j_config = config_provider.get_neo4j_config()
        with GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        ) as driver:
            with driver.session() as session:
                result = session.run(query, **params)
                return [record.data() for record in result]

    @staticmethod
    def process_query_result(result: List[Dict[str, Any]], repo_name: str, node_name: str) -> Dict[str, Any]:
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