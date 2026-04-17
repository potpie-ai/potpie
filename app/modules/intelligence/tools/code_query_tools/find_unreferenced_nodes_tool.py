import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class FindUnreferencedNodesInput(BaseModel):
    project_id: str = Field(..., description="The repository ID (UUID)")
    node_types: List[str] = Field(
        default=["FUNCTION", "CLASS", "INTERFACE"],
        description="Node types to check. Defaults to FUNCTION, CLASS, INTERFACE.",
    )
    limit: int = Field(
        default=200,
        description="Maximum number of unreferenced nodes to return (default 200).",
    )


class FindUnreferencedNodesTool:
    """Tool for finding nodes in the knowledge graph that have no incoming edges.

    A node with no incoming REFERENCES edges from other code nodes is a candidate
    for dead code — it is never called or used by anything else in the codebase.
    Entry-point detection (main, routes, tests, exports) is left to the agent.
    """

    name = "find_unreferenced_nodes"
    description = """Queries the knowledge graph for FUNCTION, CLASS, and INTERFACE nodes
    that have zero incoming edges from other code nodes — i.e. nothing in the codebase
    calls or references them. Returns node_id, name, file_path, start_line, end_line,
    and type for each candidate. Use this as the first step in dead code analysis;
    then filter out legitimate entry points (main, API routes, test functions, exports).

    :param project_id: string, the repository ID (UUID).
    :param node_types: array of strings, node types to include (default: FUNCTION, CLASS, INTERFACE).
    :param limit: int, maximum results to return (default: 200).

    example:
    {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "node_types": ["FUNCTION", "CLASS"],
        "limit": 100
    }
    """

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
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
        node_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_types, limit)

    def run(
        self,
        project_id: str,
        node_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        if node_types is None:
            node_types = ["FUNCTION", "CLASS", "INTERFACE"]
        try:
            nodes = self._find_unreferenced(project_id, node_types, limit)
            return {
                "unreferenced_nodes": nodes,
                "count": len(nodes),
                "note": (
                    "These nodes have no incoming edges from other code nodes. "
                    "Filter out entry points (main, API routes, __init__, test functions, "
                    "exported symbols) before treating them as dead code."
                ),
            }
        except Exception as e:
            logger.exception("find_unreferenced_nodes failed for project %s", project_id)
            return {"error": str(e)}

    def _find_unreferenced(
        self, project_id: str, node_types: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        # IS_LEAF edges are structural (added by the graph builder to mark leaf nodes)
        # and do not represent actual code references. We exclude them so that
        # structural containment doesn't mask truly unreferenced nodes.
        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE n.type IN $node_types
        OPTIONAL MATCH (caller:NODE {repoId: $project_id})-[r]->(n)
        WHERE type(r) <> 'IS_LEAF'
        WITH n, COUNT(caller) AS caller_count
        WHERE caller_count = 0
        RETURN
            n.node_id   AS node_id,
            n.name      AS name,
            n.file_path AS file_path,
            n.start_line AS start_line,
            n.end_line   AS end_line,
            n.type       AS type
        ORDER BY n.file_path, n.start_line
        LIMIT $limit
        """
        with self.neo4j_driver.session() as session:
            result = session.run(
                query,
                project_id=project_id,
                node_types=node_types,
                limit=limit,
            )
            return [
                {
                    "node_id": record["node_id"],
                    "name": record["name"],
                    "file_path": self._relative_path(record["file_path"]),
                    "start_line": record["start_line"],
                    "end_line": record["end_line"],
                    "type": record["type"],
                }
                for record in result
            ]

    @staticmethod
    def _relative_path(file_path: str) -> str:
        if not file_path or file_path == "Unknown":
            return "Unknown"
        parts = file_path.split("/")
        try:
            idx = parts.index("projects")
            return "/".join(parts[idx + 2 :])
        except ValueError:
            return file_path

    def close(self) -> None:
        if hasattr(self, "neo4j_driver") and self.neo4j_driver is not None:
            try:
                self.neo4j_driver.close()
            except Exception:
                pass
            self.neo4j_driver = None

    def __del__(self):
        if hasattr(self, "neo4j_driver") and self.neo4j_driver is not None:
            try:
                self.neo4j_driver.close()
            except Exception:
                pass
            self.neo4j_driver = None


def find_unreferenced_nodes_tool(sql_db: Session) -> StructuredTool:
    tool_instance = FindUnreferencedNodesTool(sql_db)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Find Unreferenced Nodes",
        description=(
            "Queries the knowledge graph for FUNCTION, CLASS, and INTERFACE nodes with "
            "zero incoming edges — nothing in the codebase calls or references them. "
            "Returns file path, line numbers, and node type for each candidate. "
            "Use as the starting point for dead code analysis; the agent should then "
            "filter out entry points, exports, test helpers, and framework hooks."
        ),
        args_schema=FindUnreferencedNodesInput,
    )
