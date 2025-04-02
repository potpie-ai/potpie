import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import StructuredTool
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from sqlalchemy.orm import Session


class NodeRelevance(BaseModel):
    """Model for evaluating node relevance for integration test context"""

    node_id: str = Field(..., description="The ID of the node being evaluated")
    node_name: str = Field(..., description="The name of the node being evaluated")
    is_relevant: bool = Field(
        ..., description="Whether this node is relevant for integration test context"
    )
    relevance_score: float = Field(
        ..., description="Relevance score between 0-1 where 1 is highly relevant"
    )
    reason: str = Field(..., description="Reasoning for the relevance determination")


class RelevantNode(BaseModel):
    """Model for a node in the intelligently filtered graph"""

    id: str
    name: str
    type: str
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    relevance_score: float
    reason: str
    children: List["RelevantNode"] = []


class IntelligentCodeGraphTool:
    name = "intelligent_code_graph"
    description = """Intelligently fetches a code graph starting from a node ID, filtering out
    irrelevant nodes that add noise to the context. The filtering is done using rule-based
    evaluation to identify relevant nodes for integration test generation.

    :param project_id: string, the repository ID (UUID).
    :param node_id: string, the ID of the node to retrieve the graph for.
    :param relevance_threshold: float, optional, minimum relevance score for a node to be included (default: 0.6).
    :param max_depth: integer, optional, maximum depth of relationships to traverse (default: 5).

    example:
    {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "node_id": "123e4567-e89b-12d3-a456-426614174000",
        "relevance_threshold": 0.7,
        "max_depth": 4
    }
    """

    def __init__(
        self, sql_db: Session, provider_service: ProviderService, user_id: str
    ):
        self.sql_db = sql_db
        self.user_id = user_id
        self.code_graph_tool = GetCodeGraphFromNodeIdTool(sql_db)
        self.code_from_node_tool = GetCodeFromNodeIdTool(sql_db, user_id)
        self.visited_nodes: Set[str] = set()

    def run(
        self,
        project_id: str = None,
        node_id: str = None,
        relevance_threshold: float = 0.6,
        max_depth: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async implementation in an event loop"""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                with ThreadPoolExecutor() as pool:
                    return pool.submit(
                        lambda: asyncio.run(
                            self.arun(
                                project_id, node_id, relevance_threshold, max_depth
                            )
                        )
                    ).result()
        except RuntimeError:
            return asyncio.run(
                self.arun(project_id, node_id, relevance_threshold, max_depth)
            )

    async def arun(
        self,
        project_id: str = None,
        node_id: str = None,
        relevance_threshold: float = 0.6,
        max_depth: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async version of run"""
        if isinstance(project_id, dict):
            params = project_id
            project_id = params.get("project_id")
            node_id = params.get("node_id")
            relevance_threshold = params.get("relevance_threshold", 0.6)
            max_depth = params.get("max_depth", 5)

        if not project_id or not node_id:
            return {
                "error": "Missing required parameters: project_id and node_id must be provided"
            }

        try:
            self.visited_nodes = set()

            result = self.code_graph_tool.run(project_id, node_id, max_depth=1)
            if "error" in result:
                return result

            if not result.get("graph") or not result["graph"].get("root_node"):
                return {"error": f"Invalid graph structure returned for node {node_id}"}

            root_node = result["graph"]["root_node"]
            if (
                not isinstance(root_node, dict)
                or "id" not in root_node
                or "name" not in root_node
            ):
                return {"error": f"Invalid root node structure for node {node_id}"}

            code_result = await asyncio.to_thread(
                self.code_from_node_tool.run, project_id, node_id
            )
            root_code = code_result.get("code", "")

            try:
                filtered_graph = await self._process_node_recursively_async(
                    project_id=project_id,
                    node=root_node,
                    node_code=root_code,
                    relevance_threshold=relevance_threshold,
                    current_depth=0,
                    max_depth=max_depth,
                )
            except Exception as e:
                logging.warning(
                    f"Node processing failed: {str(e)}; using simplified filtering"
                )
                filtered_graph = self._create_relevant_node(
                    root_node, 1.0, "Root node (simplified processing due to error)"
                )

                child_nodes = []

                async def process_child(child):
                    relevance = 0.9
                    reason = "Basic relevance assessment"

                    node_name = child.get("name", "").lower()
                    if any(
                        term in node_name for term in ["log", "debug", "print", "test"]
                    ):
                        relevance = 0.4
                        reason = "Likely utility or debug code"

                    if relevance >= relevance_threshold:
                        return self._create_relevant_node(child, relevance, reason)
                    return None

                tasks = [
                    process_child(child) for child in root_node.get("children", [])
                ]
                results = await asyncio.gather(*tasks)
                filtered_graph["children"] = [r for r in results if r is not None]

            return {
                "graph": {
                    "name": result["graph"]["name"],
                    "repo_name": result["graph"]["repo_name"],
                    "branch_name": result["graph"].get("branch_name", ""),
                    "root_node": filtered_graph,
                    "nodes_evaluated": len(self.visited_nodes),
                    "nodes_included": self._count_nodes(filtered_graph),
                }
            }
        except Exception as e:
            logging.exception(f"Error in intelligent code graph tool: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    async def _process_node_recursively_async(
        self,
        project_id: str,
        node: Dict[str, Any],
        node_code: str,
        relevance_threshold: float,
        current_depth: int,
        max_depth: int,
    ) -> Dict[str, Any]:
        """Process nodes recursively with parallel evaluation"""
        node_id = node["id"]
        self.visited_nodes.add(node_id)

        if current_depth >= max_depth:
            return self._create_relevant_node(node, 1.0, "Maximum depth reached")

        if not node.get("children") or len(node["children"]) == 0:
            return self._create_relevant_node(
                node, 1.0, "Leaf node - no further evaluation needed"
            )

        # Evaluate children in parallel
        evaluations = await self._evaluate_nodes_async(node["children"])

        # Process relevant children in parallel
        async def process_child(child, evaluation):
            if evaluation.relevance_score >= relevance_threshold:
                child_code_result = await asyncio.to_thread(
                    self.code_from_node_tool.run, project_id, child["id"]
                )
                child_code = child_code_result.get("code", "")

                processed_child = await self._process_node_recursively_async(
                    project_id=project_id,
                    node=child,
                    node_code=child_code,
                    relevance_threshold=relevance_threshold,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                )

                processed_child["relevance_score"] = evaluation.relevance_score
                processed_child["reason"] = evaluation.reason
                return processed_child
            return None

        tasks = [
            process_child(child, eval_)
            for child, eval_ in zip(node["children"], evaluations)
        ]
        processed_children = await asyncio.gather(*tasks)
        relevant_children = [c for c in processed_children if c is not None]

        processed_node = self._create_relevant_node(
            node, 1.0, "Entry point for analysis"
        )
        processed_node["children"] = relevant_children

        return processed_node

    async def _evaluate_nodes_async(
        self, children: List[Dict[str, Any]]
    ) -> List[NodeRelevance]:
        """Evaluate nodes based on rule-based relevance criteria"""
        try:
            batch_size = 10
            batches = [
                children[i : i + batch_size]
                for i in range(0, len(children), batch_size)
            ]

            async def process_batch(batch):
                evaluations = []
                for child in batch:
                    node_name = child.get("name", "").lower()

                    relevance_score = 0.7
                    is_relevant = True
                    reason = "Possibly relevant based on name/type"

                    # Low relevance patterns (debug, test, etc.)
                    if any(
                        term in node_name
                        for term in [
                            "log",
                            "debug",
                            "print",
                            "mock",
                            "fixture",
                            "stub",
                            "fake",
                            "console",
                            ".test.",
                            ".spec.",
                            "test_",
                            "spec_",
                        ]
                    ):
                        relevance_score = 0.3
                        is_relevant = False
                        reason = "Likely utility, debug or test code"

                    # High relevance backend patterns
                    elif any(
                        term in node_name
                        for term in [
                            "api",
                            "service",
                            "controller",
                            "repository",
                            "dao",
                            "database",
                            "model",
                            "entity",
                            "schema",
                            "resolver",
                            "middleware",
                            "handler",
                            "route",
                            "endpoint",
                            "graphql",
                            "query",
                            "mutation",
                        ]
                    ):
                        relevance_score = 0.9
                        is_relevant = True
                        reason = "Core backend business logic or integration point"

                    # High relevance frontend patterns
                    elif any(
                        term in node_name
                        for term in [
                            "component",
                            "page",
                            "view",
                            "screen",
                            "layout",
                            "template",
                            "form",
                            "hook",
                            "store",
                            "reducer",
                            "action",
                            "context",
                            "provider",
                            "client",
                            "app",
                            "router",
                        ]
                    ):
                        relevance_score = 0.9
                        is_relevant = True
                        reason = "Core frontend component or state management"

                    # Medium relevance support code
                    elif any(
                        term in node_name
                        for term in [
                            "util",
                            "helper",
                            "factory",
                            "builder",
                            "migration",
                            "config",
                            "setup",
                            "constant",
                            "type",
                            "interface",
                            "style",
                            "theme",
                            "asset",
                            "locale",
                            "i18n",
                            "validation",
                            "formatter",
                            "transform",
                        ]
                    ):
                        relevance_score = 0.6
                        is_relevant = True
                        reason = "Support code that may be needed for integration"

                    evaluations.append(
                        NodeRelevance(
                            node_id=child["id"],
                            node_name=child["name"],
                            is_relevant=is_relevant,
                            relevance_score=relevance_score,
                            reason=reason,
                        )
                    )
                return evaluations

            tasks = [process_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*tasks)

            return [eval_ for batch_result in batch_results for eval_ in batch_result]

        except Exception as e:
            logging.exception(f"Error evaluating nodes: {str(e)}")
            return [
                NodeRelevance(
                    node_id=child["id"],
                    node_name=child["name"],
                    is_relevant=True,
                    relevance_score=0.8,
                    reason="Evaluation failed, defaulting to including node",
                )
                for child in children
            ]

    def _create_relevant_node(
        self, node: Dict[str, Any], relevance_score: float, reason: str
    ) -> Dict[str, Any]:
        """Create a node with relevance information"""
        relevant_node = {
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "relevance_score": relevance_score,
            "reason": reason,
            "children": [],
        }

        for field in ["file_path", "start_line", "end_line", "relationship"]:
            if field in node:
                relevant_node[field] = node[field]

        return relevant_node

    def _count_nodes(self, node: Dict[str, Any]) -> int:
        """Count the number of nodes in the filtered graph"""
        count = 1  # Count this node
        for child in node.get("children", []):
            count += self._count_nodes(child)
        return count


def get_intelligent_code_graph_tool(
    sql_db: Session, provider_service: ProviderService, user_id: str
) -> StructuredTool:
    """Create and return the intelligent code graph tool"""
    tool = IntelligentCodeGraphTool(sql_db, provider_service, user_id)

    class IntelligentCodeGraphSchema(BaseModel):
        project_id: str = Field(..., description="The repository ID (UUID).")
        node_id: str = Field(
            ..., description="The ID of the node to retrieve the graph for."
        )
        relevance_threshold: float = Field(
            0.6,
            description="Minimum relevance score for a node to be included (default: 0.6).",
        )
        max_depth: int = Field(
            5, description="Maximum depth of relationships to traverse (default: 5)."
        )

    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=IntelligentCodeGraphSchema,
    )
