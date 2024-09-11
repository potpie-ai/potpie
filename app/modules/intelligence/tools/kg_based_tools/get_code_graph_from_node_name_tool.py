from typing import Any, Dict
from neo4j import GraphDatabase
from sqlalchemy.orm import Session
from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project

class GetCodeGraphFromNodeIdTool:
    name = "get_code_graph_from_node_id"
    description = "Retrieves a code graph for a specific node in a repository given its node ID"

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
            project = self._get_project(repo_id)
            if not project:
                return {"error": f"Project with ID '{repo_id}' not found in database"}

            graph_data = self._get_graph_data(repo_id, node_id)
            if not graph_data:
                return {"error": f"No graph data found for node ID '{node_id}' in repo '{repo_id}'"}

            return self._process_graph_data(graph_data, project)
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_project(self, repo_id: str) -> Project:
        return self.sql_db.query(Project).filter(Project.id == repo_id).first()

    def _get_graph_data(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        query = """
        MATCH (start:NODE {node_id: $node_id, repoId: $repo_id})
        CALL apoc.path.subgraphAll(start, {
            relationshipFilter: "CALLS|IMPLEMENTS|INHERITS|CONTAINS|REFERENCES>",
            maxLevel: 2
        })
        YIELD nodes, relationships
        WITH nodes, relationships
        UNWIND nodes as node
        OPTIONAL MATCH (node)-[r]->(related)
        WHERE related IN nodes
        WITH node, collect({
            target: related.node_id,
            type: type(r),
            properties: properties(r)
        }) as outgoing_edges
        RETURN collect({
            id: node.node_id,
            label: node.name,
            metadata: {
                type: head(labels(node)),
                file_path: node.file_path,
                start_line: node.start_line,
                end_line: node.end_line
            },
            edges: outgoing_edges
        }) as nodes
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, repo_id=repo_id)
            record = result.single()
            if not record:
                return None
            return {"nodes": record["nodes"]}

    def _process_graph_data(self, graph_data: Dict[str, Any], project: Project) -> Dict[str, Any]:
        nodes = {}
        for node in graph_data["nodes"]:
            node_id = node["id"]
            nodes[node_id] = {
                "label": node["label"],
                "metadata": {
                    "type": node["metadata"]["type"],
                    "file_path": self._get_relative_file_path(node["metadata"]["file_path"]),
                    "start_line": node["metadata"]["start_line"],
                    "end_line": node["metadata"]["end_line"]
                },
                "edges": node["edges"]
            }

        return {
            "graph": {
                "directed": True,
                "type": "Code Graph",
                "label": f"Code Graph for {project.repo_name}",
                "metadata": {
                    "repo_name": project.repo_name,
                    "branch_name": project.branch_name
                },
                "nodes": nodes
            }
        }

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        if not file_path or file_path == "Unknown":
            return "Unknown"
        parts = file_path.split("/")
        try:
            projects_index = parts.index("projects")
            return "/".join(parts[projects_index + 2 :])
        except ValueError:
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()

    async def arun(self, repo_id: str, node_id: str) -> Dict[str, Any]:
        return self.run(repo_id, node_id)

from typing import Any, Dict
from neo4j import GraphDatabase
from sqlalchemy.orm import Session
from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project

class GetCodeGraphFromNodeNameTool:
    name = "get_code_graph_from_node_name"
    description = "Retrieves a code graph for a specific node in a repository given its node name"

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    def run(self, repo_id: str, node_name: str) -> Dict[str, Any]:
        print(f"Running get code graph from node name tool for repo_id: {repo_id}")
        try:
            project = self._get_project(repo_id)
            if not project:
                return {"error": f"Project with ID '{repo_id}' not found in database"}

            graph_data = self._get_graph_data(repo_id, node_name)
            if not graph_data:
                return {"error": f"No graph data found for node name '{node_name}' in repo '{repo_id}'"}

            return self._process_graph_data(graph_data, project)
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_project(self, repo_id: str) -> Project:
        project = self.sql_db.query(Project).filter(Project.id == repo_id).first()
        print(f"Getting project for repo_id {repo_id}: {project}")
        return project

    def _get_graph_data(self, repo_id: str, node_name: str) -> Dict[str, Any]:
        query = """
        MATCH (start:NODE {repoId: $repo_id})
        WHERE toLower(start.name) = toLower($node_name)
        CALL apoc.path.subgraphAll(start, {
            relationshipFilter: "CALLS|IMPLEMENTS|INHERITS|CONTAINS|REFERENCES>",
            maxLevel: 2
        })
        YIELD nodes, relationships
        WITH nodes, relationships
        UNWIND nodes as node
        OPTIONAL MATCH (node)-[r]->(related)
        WHERE related IN nodes
        WITH node, collect({
            target: related.node_id,
            type: type(r),
            properties: properties(r)
        }) as outgoing_edges
        RETURN collect({
            id: node.node_id,
            label: node.name,
            metadata: {
                type: head(labels(node)),
                file_path: node.file_path,
                start_line: node.start_line,
                end_line: node.end_line
            },
            edges: outgoing_edges
        }) as nodes
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_name=node_name, repo_id=repo_id)
            record = result.single()
            if not record:
                return None
            return {"nodes": record["nodes"]}

    def _process_graph_data(self, graph_data: Dict[str, Any], project: Project) -> Dict[str, Any]:
        nodes = {}
        for node in graph_data["nodes"]:
            node_id = node["id"]
            nodes[node_id] = {
                "label": node["label"],
                "metadata": {
                    "type": node["metadata"]["type"],
                    "file_path": self._get_relative_file_path(node["metadata"]["file_path"]),
                    "start_line": node["metadata"]["start_line"],
                    "end_line": node["metadata"]["end_line"]
                },
                "edges": node["edges"]
            }

        return {
            "graph": {
                "directed": True,
                "type": "Code Graph",
                "label": f"Code Graph for {project.repo_name}",
                "metadata": {
                    "repo_name": project.repo_name,
                    "branch_name": project.branch_name
                },
                "nodes": nodes
            }
        }

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        if not file_path or file_path == "Unknown":
            return "Unknown"
        parts = file_path.split("/")
        try:
            projects_index = parts.index("projects")
            return "/".join(parts[projects_index + 2 :])
        except ValueError:
            return file_path

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()

    async def arun(self, repo_id: str, node_name: str) -> Dict[str, Any]:
        return self.run(repo_id, node_name)