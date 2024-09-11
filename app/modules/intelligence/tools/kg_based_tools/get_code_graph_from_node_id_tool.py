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
        CALL apoc.path.spanningTree(start, {
            relationshipFilter: "CONTAINS|CALLS|FUNCTION_DEFINITION|IMPORTS|INSTANTIATES|CLASS_DEFINITION>",
            maxLevel: 10
        })
        YIELD path
        WITH nodes(path) AS nodePath, relationships(path) AS rels
        UNWIND nodePath AS node
        OPTIONAL MATCH (node)-[r]->(child:NODE)
        WHERE child IN nodePath AND type(r) <> 'IS_LEAF'
        WITH node, collect({
            id: child.node_id,
            name: child.name,
            type: head(labels(child)),
            file_path: child.file_path,
            start_line: child.start_line,
            end_line: child.end_line
        }) as children
        RETURN {
            id: node.node_id,
            name: node.name,
            type: head(labels(node)),
            file_path: node.file_path,
            start_line: node.start_line,
            end_line: node.end_line,
            children: children
        } as node_data
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, repo_id=repo_id)
            nodes = [record["node_data"] for record in result]
            if not nodes:
                return None
            return self._build_tree(nodes, node_id)

    def _build_tree(self, nodes, root_id):
        node_map = {node['id']: node for node in nodes}
        root = node_map.get(root_id)
        if not root:
            return None

        visited = set()

        def build_node_tree(current_node):
            if current_node['id'] in visited:
                return None
            visited.add(current_node['id'])

            current_node['children'] = [
                child for child in current_node['children'] if child['id'] in node_map
            ]

            for child in current_node['children']:
                child_node = node_map[child['id']]
                built_child = build_node_tree(child_node)
                if built_child:
                    child['children'] = built_child['children']
                else:
                    current_node['children'].remove(child)

            return current_node

        return build_node_tree(root)

    def _process_graph_data(self, graph_data: Dict[str, Any], project: Project) -> Dict[str, Any]:
        def process_node(node):
            processed_node = {
                "id": node["id"],
                "name": node["name"],
                "type": node["type"],
                "file_path": self._get_relative_file_path(node["file_path"]),
                "start_line": node["start_line"],
                "end_line": node["end_line"],
                "function_calls": node.get("function_calls", []),
                "signature": node.get("signature", ""),
                "children": []
            }
            for child in node.get("children", []):
                processed_child = process_node(child)
                processed_child["relationship"] = child["relationship"]
                processed_node["children"].append(processed_child)
            return processed_node

        root_node = process_node(graph_data)

        return {
            "graph": {
                "name": f"Code Graph for {project.repo_name}",
                "repo_name": project.repo_name,
                "branch_name": project.branch_name,
                "root_node": root_node
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
