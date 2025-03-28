import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project


class GetCodeGraphFromNodeIdTool:
    """Tool for retrieving a code graph for a specific node in a repository given its node ID."""

    name = "get_code_graph_from_node_id"
    description = """Retrieves a code graph showing relationships between nodes starting from a specific node ID.
        :param project_id: string, the repository ID (UUID).
        :param node_id: string, the ID of the node to retrieve the graph for (UUID).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000"
            }

        Returns dictionary containing:
        - graph: {
            name: string - name of the graph
            repo_name: string - repository name
            branch_name: string - branch name
            root_node: object - hierarchical structure of nodes with relationships
          }
        """

    def __init__(self, sql_db: Session):
        """
        Initialize the tool with a SQL database session.

        Args:
            sql_db (Session): SQLAlchemy database session.
        """
        self.sql_db = sql_db
        self.neo4j_driver = self._create_neo4j_driver()

    def _create_neo4j_driver(self) -> GraphDatabase.driver:
        """Create and return a Neo4j driver instance."""
        neo4j_config = config_provider.get_neo4j_config()
        return GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    async def arun(self, project_id: str, node_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_id)

    def run(self, project_id: str, node_id: str) -> Dict[str, Any]:
        """
        Run the tool to retrieve the code graph.

        Args:
            project_id (str): Repository ID.
            node_id (str): ID of the node to retrieve the graph for.

        Returns:
            Dict[str, Any]: Code graph data or error message.
        """
        try:
            project = self._get_project(project_id)
            if not project:
                return {
                    "error": f"Project with ID '{project_id}' not found in database"
                }

            graph_data = self._get_graph_data(project_id, node_id)
            if not graph_data:
                return {
                    "error": f"No graph data found for node ID '{node_id}' in repo '{project_id}'"
                }

            return self._process_graph_data(graph_data, project)
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_project(self, project_id: str) -> Optional[Project]:
        """Retrieve project from the database."""
        return self.sql_db.query(Project).filter(Project.id == project_id).first()

    def _get_graph_data(
        self, project_id: str, node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve graph data from Neo4j."""
        query = """
        MATCH (start:NODE {node_id: $node_id, repoId: $project_id})
        CALL apoc.path.subgraphAll(start, {
            maxLevel: 10
        })
        YIELD nodes, relationships
        UNWIND nodes AS node
        OPTIONAL MATCH (node)-[r]->(child:NODE)
        WHERE child IN nodes AND type(r) <> 'IS_LEAF'
        WITH node, collect({
            id: child.node_id,
            name: child.name,
            type: head(labels(child)),
            file_path: child.file_path,
            start_line: child.start_line,
            end_line: child.end_line,
            relationship: type(r)
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
            result = session.run(query, node_id=node_id, project_id=project_id)
            nodes = [record["node_data"] for record in result]
            if not nodes:
                return None
            return self._build_tree(nodes, node_id)

    def _build_tree(
        self, nodes: List[Dict[str, Any]], root_id: str
    ) -> Optional[Dict[str, Any]]:
        """Build a tree structure from the graph data."""
        node_map = {node["id"]: node for node in nodes}
        root = node_map.get(root_id)
        if not root:
            return None

        visited = set()

        def build_node_tree(current_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if current_node["id"] in visited:
                return None
            visited.add(current_node["id"])

            current_node["children"] = [
                child for child in current_node["children"] if child["id"] in node_map
            ]

            for child in current_node["children"]:
                child_node = node_map[child["id"]]
                built_child = build_node_tree(child_node)
                if built_child:
                    child["children"] = built_child["children"]
                else:
                    current_node["children"].remove(child)

            return current_node

        return build_node_tree(root)

    def _process_graph_data(
        self, graph_data: Dict[str, Any], project: Project
    ) -> Dict[str, Any]:
        """Process the graph data and prepare the final output."""

        def process_node(node: Dict[str, Any]) -> Dict[str, Any]:
            processed_node = {
                "id": node["id"],
                "name": node["name"],
                "type": node["type"],
                "file_path": self._get_relative_file_path(node["file_path"]),
                "start_line": node["start_line"],
                "end_line": node["end_line"],
                "children": [],
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
                "root_node": root_node,
            }
        }

    @staticmethod
    def _get_relative_file_path(file_path: str) -> str:
        """Convert absolute file path to relative path."""
        if not file_path or file_path == "Unknown":
            return "Unknown"
        parts = file_path.split("/")
        try:
            projects_index = parts.index("projects")
            return "/".join(parts[projects_index + 2 :])
        except ValueError:
            return file_path

    def __del__(self):
        """Ensure Neo4j driver is closed when the object is destroyed."""
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_code_graph_from_node_id_tool(sql_db: Session) -> StructuredTool:
    tool_instance = GetCodeGraphFromNodeIdTool(sql_db)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Get Code Graph From Node ID",
        description="""Retrieves a code graph showing relationships between nodes starting from a specific node ID.
        :param project_id: string, the repository ID (UUID).
        :param node_id: string, the ID of the node to retrieve the graph for (UUID).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000"
            }

        Returns dictionary containing:
        - graph: {
            name: string - name of the graph
            repo_name: string - repository name
            branch_name: string - branch name
            root_node: object - hierarchical structure of nodes with relationships
          }
        """,
    )
