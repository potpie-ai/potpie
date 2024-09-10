from typing import Dict, Any
from neo4j import GraphDatabase
from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project
from sqlalchemy.orm import Session

class GetCodeFlowFromNodeIdTool:
    name = "get_code_flow_from_node_id"
    description = "Generates a code flow graph based on a given node ID, focusing on outbound relationships"

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        )

    def run(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        print(f"GetCodeFlowFromNodeIdTool.run called with repo_id: {repo_id}, node_id: {node_id}")
        try:
            project = self._get_project(repo_id)
            if not project:
                return {"error": f"Project with ID '{repo_id}' not found in database"}

            graph_data = self._get_code_flow(repo_id, node_id)
            if not graph_data:
                return {"error": f"No outbound code flow found for node ID '{node_id}' in repo '{repo_id}'"}

            code_flow = self._process_graph_data(graph_data, project)
            return {"code_flow": code_flow}
        except Exception as e:
            print(f"Error in GetCodeFlowFromNodeIdTool.run: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_project(self, repo_id: str) -> Project:
        return self.sql_db.query(Project).filter(Project.id == repo_id).first()

    def _get_code_flow(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (start:NODE {node_id: $node_id, repoId: $repo_id})
        CALL apoc.path.subgraphAll(start, {
            relationshipFilter: "CALLS|IMPLEMENTS|INHERITS|CONTAINS|REFERENCES>",
            maxLevel: 2
        })
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, repo_id=repo_id)
            record = result.single()
            if not record:
                return None
            return {
                "nodes": record["nodes"],
                "relationships": record["relationships"]
            }

    def _process_graph_data(self, graph_data: Dict[str, Any], project: Project) -> Dict[str, Any]:
        nodes = []
        for node in graph_data["nodes"]:
            node_data = {
                "id": node["node_id"],
                "name": node.get("name", "Unknown"),
                "type": list(node.labels)[0] if node.labels else "Unknown",
                "file_path": self._get_relative_file_path(node.get("file_path", "Unknown")),
                "start_line": node.get("start_line", -1),
                "end_line": node.get("end_line", -1)
            }
            nodes.append(node_data)

        relationships = [
            {
                "source": rel.start_node["node_id"],
                "target": rel.end_node["node_id"],
                "type": type(rel).__name__
            } for rel in graph_data["relationships"]
        ]

        return {"nodes": nodes, "relationships": relationships}

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        if not file_path or file_path == "Unknown":
            return "Unknown"
        parts = file_path.split('/')
        try:
            projects_index = parts.index('projects')
            return '/'.join(parts[projects_index + 2:])
        except ValueError:
            print(f"'projects' not found in file path: {file_path}")
            return file_path

    def __del__(self):
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

    async def arun(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        return self.run(repo_id, node_id)
