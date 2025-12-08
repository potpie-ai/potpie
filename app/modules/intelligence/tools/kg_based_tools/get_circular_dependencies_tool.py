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


class GetCircularDependenciesInput(BaseModel):
    project_id: str = Field(description="The project ID, this is a UUID")
    node_id: Optional[str] = Field(
        default=None,
        description="Optional: specific node ID to check for cycles involving this node",
    )
    max_cycle_length: int = Field(
        default=5, description="Maximum cycle length to detect (2-10)"
    )
    limit: int = Field(
        default=20, description="Maximum number of cycles to return (1-50)"
    )


class GetCircularDependenciesTool:
    name = "Get Circular Dependencies"
    description = """Detects circular dependencies (cycles) in the codebase.
        A circular dependency occurs when node A depends on B, B depends on C,
        and C depends back on A (or any similar cycle).

        This helps identify:
        - Tight coupling between components
        - Potential refactoring opportunities
        - Architecture issues that may cause problems

        :param project_id: string, the project ID (UUID).
        :param node_id: string, optional - check cycles involving this specific node.
        :param max_cycle_length: integer, maximum cycle length 2-10 (default: 5).
        :param limit: integer, max cycles to return 1-50 (default: 20).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": null,
                "max_cycle_length": 5,
                "limit": 20
            }

        Returns list of detected cycles with all nodes involved.
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
        node_id: Optional[str] = None,
        max_cycle_length: int = 5,
        limit: int = 20,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.run, project_id, node_id, max_cycle_length, limit
        )

    def run(
        self,
        project_id: str,
        node_id: Optional[str] = None,
        max_cycle_length: int = 5,
        limit: int = 20,
    ) -> Dict[str, Any]:
        try:
            project = asyncio.run(
                ProjectService(self.sql_db).get_project_repo_details_from_db(
                    project_id, self.user_id
                )
            )
            if not project:
                raise ValueError(
                    f"Project with ID '{project_id}' not found for user '{self.user_id}'"
                )

            max_cycle_length = max(2, min(10, max_cycle_length))
            limit = max(1, min(50, limit))

            if node_id:
                cycles = self._find_cycles_for_node(
                    project_id, node_id, max_cycle_length, limit
                )
            else:
                cycles = self._find_all_cycles(project_id, max_cycle_length, limit)

            if not cycles:
                return {
                    "project_id": project_id,
                    "cycles": [],
                    "cycle_count": 0,
                    "message": "No circular dependencies found",
                }

            file_cycles = self._group_cycles_by_file(cycles)

            return {
                "project_id": project_id,
                "node_id": node_id,
                "cycles": cycles,
                "cycle_count": len(cycles),
                "file_level_cycles": file_cycles,
                "file_cycle_count": len(file_cycles),
            }

        except Exception as e:
            logger.error(f"Error in GetCircularDependenciesTool: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}

    def _find_all_cycles(
        self, project_id: str, max_cycle_length: int, limit: int
    ) -> List[Dict[str, Any]]:
        query = (
            """
        MATCH path = (n:NODE {repoId: $project_id})-[:CALLS|REFERENCES*2..%d]->(n)
        WITH path, nodes(path) AS pathNodes
        WHERE ALL(node IN pathNodes WHERE node.repoId = $project_id)
        WITH pathNodes,
             [node IN pathNodes | node.node_id] AS nodeIds,
             size(pathNodes) AS cycleLength
        WITH pathNodes, nodeIds, cycleLength,
             head(apoc.coll.sort(nodeIds)) AS canonicalStart
        WHERE nodeIds[0] = canonicalStart
        RETURN DISTINCT
            [n IN pathNodes | {
                node_id: n.node_id,
                name: n.name,
                file_path: n.file_path,
                start_line: n.start_line,
                end_line: n.end_line,
                labels: labels(n)
            }] AS cycle_nodes,
            cycleLength - 1 AS cycle_length
        ORDER BY cycle_length
        LIMIT $limit
        """
            % max_cycle_length
        )

        fallback_query = (
            """
        MATCH path = (n:NODE {repoId: $project_id})-[:CALLS|REFERENCES*2..%d]->(n)
        WITH path, nodes(path) AS pathNodes
        WHERE ALL(node IN pathNodes WHERE node.repoId = $project_id)
        WITH DISTINCT pathNodes, size(pathNodes) - 1 AS cycleLength
        RETURN
            [n IN pathNodes | {
                node_id: n.node_id,
                name: n.name,
                file_path: n.file_path,
                start_line: n.start_line,
                end_line: n.end_line,
                labels: labels(n)
            }] AS cycle_nodes,
            cycleLength AS cycle_length
        ORDER BY cycle_length
        LIMIT $limit
        """
            % max_cycle_length
        )

        with self.neo4j_driver.session() as session:
            try:
                result = session.run(query, project_id=project_id, limit=limit)
                return self._process_cycle_results(result)
            except Exception:
                result = session.run(fallback_query, project_id=project_id, limit=limit)
                return self._process_cycle_results(result)

    def _find_cycles_for_node(
        self, project_id: str, node_id: str, max_cycle_length: int, limit: int
    ) -> List[Dict[str, Any]]:
        query = (
            """
        MATCH (start:NODE {node_id: $node_id, repoId: $project_id})
        MATCH path = (start)-[:CALLS|REFERENCES*2..%d]->(start)
        WITH path, nodes(path) AS pathNodes
        WHERE ALL(node IN pathNodes WHERE node.repoId = $project_id)
        RETURN DISTINCT
            [n IN pathNodes | {
                node_id: n.node_id,
                name: n.name,
                file_path: n.file_path,
                start_line: n.start_line,
                end_line: n.end_line,
                labels: labels(n)
            }] AS cycle_nodes,
            size(pathNodes) - 1 AS cycle_length
        ORDER BY cycle_length
        LIMIT $limit
        """
            % max_cycle_length
        )

        with self.neo4j_driver.session() as session:
            result = session.run(
                query, node_id=node_id, project_id=project_id, limit=limit
            )
            return self._process_cycle_results(result)

    def _process_cycle_results(self, result) -> List[Dict[str, Any]]:
        cycles = []
        seen_cycles = set()

        for record in result:
            nodes = []
            node_ids = []
            for node in record["cycle_nodes"]:
                node_ids.append(node["node_id"])
                nodes.append(
                    {
                        "node_id": node["node_id"],
                        "name": node["name"],
                        "file_path": node["file_path"],
                        "start_line": node["start_line"],
                        "end_line": node["end_line"],
                        "node_type": node["labels"][0] if node["labels"] else None,
                    }
                )

            cycle_key = tuple(sorted(node_ids[:-1]))
            if cycle_key in seen_cycles:
                continue
            seen_cycles.add(cycle_key)

            cycle_path = " -> ".join([n["name"] or n["node_id"][:8] for n in nodes])

            cycles.append(
                {
                    "nodes": nodes[:-1],
                    "cycle_length": record["cycle_length"],
                    "cycle_path": cycle_path,
                }
            )

        return cycles

    def _group_cycles_by_file(
        self, cycles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        file_cycles = {}

        for cycle in cycles:
            files_in_cycle = []
            for node in cycle["nodes"]:
                file_path = node.get("file_path")
                if file_path and file_path not in files_in_cycle:
                    files_in_cycle.append(file_path)

            if len(files_in_cycle) > 1:
                cycle_key = tuple(sorted(files_in_cycle))
                if cycle_key not in file_cycles:
                    file_cycles[cycle_key] = {
                        "files": list(files_in_cycle),
                        "file_count": len(files_in_cycle),
                        "example_cycle": cycle["cycle_path"],
                    }

        return list(file_cycles.values())

    def __del__(self):
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()


def get_circular_dependencies_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = GetCircularDependenciesTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=GetCircularDependenciesInput,
    )
