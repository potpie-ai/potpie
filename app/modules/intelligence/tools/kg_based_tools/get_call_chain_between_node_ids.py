import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class GetCallChainBetweenNodesInput(BaseModel):
    project_id: str = Field(description="The project ID, this is a UUID")
    source_node_id: str = Field(description="The starting node ID, this is a UUID")
    target_node_id: str = Field(description="The destination node ID, this is a UUID")
    max_depth: int = Field(
        default=10, description="Maximum path length to search (1-15)"
    )
    find_all_paths: bool = Field(
        default=False,
        description="If true, find all paths; if false, find only the shortest path",
    )


class GetCallChainBetweenNodesTool:
    name = "Get Call Chain Between Nodes"
    description = """Finds the call chain/path between two nodes in the codebase.
        This traces CALLS relationships to show how execution flows from a source
        node to a target node.

        Use cases:
        - Understanding how function A eventually calls function B
        - Tracing the execution path between two components
        - Finding dependencies between distant parts of the codebase

        :param project_id: string, the project ID (UUID).
        :param source_node_id: string, the starting node ID.
        :param target_node_id: string, the destination node ID.
        :param max_depth: integer, maximum path length to search 1-15 (default: 10).
        :param find_all_paths: boolean, find all paths or just shortest (default: false).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "source_node_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_node_id": "987fcdeb-51a2-43e8-b6c7-123456789abc",
                "max_depth": 10,
                "find_all_paths": false
            }

        Returns the path(s) between the two nodes with all intermediate nodes.
        """

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

    async def arun(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        max_depth: int = 10,
        find_all_paths: bool = False,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.run,
            project_id,
            source_node_id,
            target_node_id,
            max_depth,
            find_all_paths,
        )

    def run(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        max_depth: int = 10,
        find_all_paths: bool = False,
    ) -> Dict[str, Any]:
        try:
            project = asyncio.run(
                ProjectService(self.sql_db).get_project_repo_details_from_db(
                    project_id, self.user_id
                )
            )
            if not project:
                raise ValueError(
                    f"Project with ID '{project_id}' not found in database for user '{self.user_id}'"
                )

            max_depth = max(1, min(15, max_depth))

            source_node = self._get_node_info(project_id, source_node_id)
            if not source_node:
                return {"error": f"Source node '{source_node_id}' not found"}

            target_node = self._get_node_info(project_id, target_node_id)
            if not target_node:
                return {"error": f"Target node '{target_node_id}' not found"}

            if source_node_id == target_node_id:
                return {
                    "source_node": source_node,
                    "target_node": target_node,
                    "paths": [],
                    "message": "Source and target are the same node",
                }

            if find_all_paths:
                paths = self._find_all_paths(
                    project_id, source_node_id, target_node_id, max_depth
                )
            else:
                paths = self._find_shortest_path(
                    project_id, source_node_id, target_node_id, max_depth
                )

            if not paths:
                return {
                    "source_node": source_node,
                    "target_node": target_node,
                    "paths": [],
                    "message": f"No call chain found between nodes within {max_depth} hops",
                }

            return {
                "project_id": project_id,
                "source_node": source_node,
                "target_node": target_node,
                "paths": paths,
                "path_count": len(paths),
                "shortest_path_length": min(len(p["nodes"]) for p in paths)
                if paths
                else 0,
            }

        except Exception as e:
            logger.error(f"Error in GetCallChainBetweenNodesTool: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}

    def _get_node_info(self, project_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
        RETURN
            n.node_id AS node_id,
            n.name AS name,
            n.file_path AS file_path,
            n.start_line AS start_line,
            n.end_line AS end_line,
            n.docstring AS docstring,
            labels(n) AS labels
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, node_id=node_id, project_id=project_id)
            record = result.single()
            if record:
                return {
                    "node_id": record["node_id"],
                    "name": record["name"],
                    "file_path": record["file_path"],
                    "start_line": record["start_line"],
                    "end_line": record["end_line"],
                    "docstring": record["docstring"],
                    "node_type": record["labels"][0] if record["labels"] else None,
                }
            return None

    def _find_shortest_path(
        self, project_id: str, source_node_id: str, target_node_id: str, max_depth: int
    ) -> List[Dict[str, Any]]:
        query = (
            """
        MATCH (source:NODE {node_id: $source_node_id, repoId: $project_id}),
              (target:NODE {node_id: $target_node_id, repoId: $project_id})
        MATCH path = shortestPath((source)-[:CALLS*1..%d]->(target))
        WITH path, nodes(path) AS pathNodes, relationships(path) AS pathRels
        RETURN
            [n IN pathNodes | {
                node_id: n.node_id,
                name: n.name,
                file_path: n.file_path,
                start_line: n.start_line,
                end_line: n.end_line,
                docstring: n.docstring,
                labels: labels(n)
            }] AS nodes,
            [r IN pathRels | {
                source: startNode(r).node_id,
                target: endNode(r).node_id
            }] AS edges,
            length(path) AS path_length
        """
            % max_depth
        )

        with self.neo4j_driver.session() as session:
            result = session.run(
                query,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                project_id=project_id,
            )
            paths = []
            for record in result:
                nodes = []
                for node in record["nodes"]:
                    nodes.append(
                        {
                            "node_id": node["node_id"],
                            "name": node["name"],
                            "file_path": node["file_path"],
                            "start_line": node["start_line"],
                            "end_line": node["end_line"],
                            "docstring": node["docstring"],
                            "node_type": node["labels"][0] if node["labels"] else None,
                        }
                    )
                paths.append(
                    {
                        "nodes": nodes,
                        "edges": record["edges"],
                        "length": record["path_length"],
                    }
                )
            return paths

    def _find_all_paths(
        self, project_id: str, source_node_id: str, target_node_id: str, max_depth: int
    ) -> List[Dict[str, Any]]:
        query = (
            """
        MATCH (source:NODE {node_id: $source_node_id, repoId: $project_id}),
              (target:NODE {node_id: $target_node_id, repoId: $project_id})
        MATCH path = (source)-[:CALLS*1..%d]->(target)
        WITH path, nodes(path) AS pathNodes, relationships(path) AS pathRels
        RETURN
            [n IN pathNodes | {
                node_id: n.node_id,
                name: n.name,
                file_path: n.file_path,
                start_line: n.start_line,
                end_line: n.end_line,
                docstring: n.docstring,
                labels: labels(n)
            }] AS nodes,
            [r IN pathRels | {
                source: startNode(r).node_id,
                target: endNode(r).node_id
            }] AS edges,
            length(path) AS path_length
        ORDER BY path_length
        LIMIT 20
        """
            % max_depth
        )

        with self.neo4j_driver.session() as session:
            result = session.run(
                query,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                project_id=project_id,
            )
            paths = []
            for record in result:
                nodes = []
                for node in record["nodes"]:
                    nodes.append(
                        {
                            "node_id": node["node_id"],
                            "name": node["name"],
                            "file_path": node["file_path"],
                            "start_line": node["start_line"],
                            "end_line": node["end_line"],
                            "docstring": node["docstring"],
                            "node_type": node["labels"][0] if node["labels"] else None,
                        }
                    )
                paths.append(
                    {
                        "nodes": nodes,
                        "edges": record["edges"],
                        "length": record["path_length"],
                    }
                )
            return paths

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_call_chain_between_node_ids(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = GetCallChainBetweenNodesTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=GetCallChainBetweenNodesInput,
    )
