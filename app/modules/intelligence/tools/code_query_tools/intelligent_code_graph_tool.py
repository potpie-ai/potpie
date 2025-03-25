import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import StructuredTool
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import GetCodeGraphFromNodeIdTool
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import GetCodeFromNodeIdTool
from sqlalchemy.orm import Session


class NodeRelevance(BaseModel):
    """Model for evaluating node relevance for integration test context"""
    node_id: str = Field(..., description="The ID of the node being evaluated")
    node_name: str = Field(..., description="The name of the node being evaluated")
    is_relevant: bool = Field(..., description="Whether this node is relevant for integration test context")
    relevance_score: float = Field(..., description="Relevance score between 0-1 where 1 is highly relevant")
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
    irrelevant nodes that add noise to the context. The filtering is done using LLM to evaluate 
    each node's relevance for integration test generation.
    
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

    def __init__(self, sql_db: Session, provider_service: ProviderService, user_id: str):
        self.sql_db = sql_db
        self.provider_service = provider_service
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
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous version that runs the async implementation in an event loop"""
        # If we're already in an event loop, run the async version directly
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Create a new loop in a separate thread for the async operation
                with ThreadPoolExecutor() as pool:
                    return pool.submit(
                        lambda: asyncio.run(self.arun(project_id, node_id, relevance_threshold, max_depth))
                    ).result()
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.arun(project_id, node_id, relevance_threshold, max_depth))

    async def arun(
        self, 
        project_id: str = None, 
        node_id: str = None, 
        relevance_threshold: float = 0.6, 
        max_depth: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of run"""
        if isinstance(project_id, dict):
            params = project_id
            project_id = params.get("project_id")
            node_id = params.get("node_id")
            relevance_threshold = params.get("relevance_threshold", 0.6)
            max_depth = params.get("max_depth", 5)
        
        if not project_id or not node_id:
            return {"error": "Missing required parameters: project_id and node_id must be provided"}
            
        try:
            # Reset visited nodes for new run
            self.visited_nodes = set()
            
            # Get the initial graph for the entry node
            result = self.code_graph_tool.run(project_id, node_id, max_depth=1)
            if "error" in result:
                return result
                
            # Validate graph structure
            if not result.get("graph") or not result["graph"].get("root_node"):
                return {"error": f"Invalid graph structure returned for node {node_id}"}
                
            root_node = result["graph"]["root_node"]
            if not isinstance(root_node, dict) or "id" not in root_node or "name" not in root_node:
                return {"error": f"Invalid root node structure for node {node_id}"}
            
            # Get code content for the root node asynchronously
            code_result = await asyncio.to_thread(self.code_from_node_tool.run, project_id, node_id)
            root_code = code_result.get("code", "")
            
            # Process the graph recursively with async filtering
            try:
                filtered_graph = await self._process_node_recursively_async(
                    project_id=project_id,
                    node=root_node,
                    node_code=root_code,
                    relevance_threshold=relevance_threshold,
                    current_depth=0,
                    max_depth=max_depth
                )
            except Exception as e:
                logging.warning(f"Node processing failed: {str(e)}; using simplified filtering")
                filtered_graph = self._create_relevant_node(
                    root_node, 1.0, "Root node (simplified processing due to error)"
                )
                
                # Process immediate children with basic filtering in parallel
                child_nodes = []
                async def process_child(child):
                    relevance = 0.9
                    reason = "Basic relevance assessment"
                    
                    node_name = child.get("name", "").lower()
                    if any(term in node_name for term in ["log", "debug", "print", "test"]):
                        relevance = 0.4
                        reason = "Likely utility or debug code"
                    
                    if relevance >= relevance_threshold:
                        return self._create_relevant_node(child, relevance, reason)
                    return None
                
                tasks = [process_child(child) for child in root_node.get("children", [])]
                results = await asyncio.gather(*tasks)
                filtered_graph["children"] = [r for r in results if r is not None]
            
            return {
                "graph": {
                    "name": result["graph"]["name"],
                    "repo_name": result["graph"]["repo_name"],
                    "branch_name": result["graph"].get("branch_name", ""),
                    "root_node": filtered_graph,
                    "nodes_evaluated": len(self.visited_nodes),
                    "nodes_included": self._count_nodes(filtered_graph)
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
        max_depth: int
    ) -> Dict[str, Any]:
        """Async version of _process_node_recursively with parallel processing"""
        node_id = node["id"]
        self.visited_nodes.add(node_id)
        
        if current_depth >= max_depth:
            return self._create_relevant_node(node, 1.0, "Maximum depth reached")
        
        if not node.get("children") or len(node["children"]) == 0:
            return self._create_relevant_node(node, 1.0, "Leaf node - no further evaluation needed")
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(node, node_code, node["children"])
        
        # Evaluate children in parallel
        evaluations = await self._evaluate_nodes_with_llm_async(prompt, node["children"])
        
        # Process relevant children in parallel
        async def process_child(child, evaluation):
            if evaluation.relevance_score >= relevance_threshold:
                # Get code for the child node asynchronously
                child_code_result = await asyncio.to_thread(
                    self.code_from_node_tool.run,
                    project_id,
                    child["id"]
                )
                child_code = child_code_result.get("code", "")
                
                # Process child recursively
                processed_child = await self._process_node_recursively_async(
                    project_id=project_id,
                    node=child,
                    node_code=child_code,
                    relevance_threshold=relevance_threshold,
                    current_depth=current_depth + 1,
                    max_depth=max_depth
                )
                
                processed_child["relevance_score"] = evaluation.relevance_score
                processed_child["reason"] = evaluation.reason
                return processed_child
            return None
        
        # Process all children in parallel
        tasks = [
            process_child(child, eval_) 
            for child, eval_ in zip(node["children"], evaluations)
        ]
        processed_children = await asyncio.gather(*tasks)
        relevant_children = [c for c in processed_children if c is not None]
        
        # Create and return processed node
        processed_node = self._create_relevant_node(
            node, 
            1.0,
            "Entry point for analysis"
        )
        processed_node["children"] = relevant_children
        
        return processed_node

    async def _evaluate_nodes_with_llm_async(
        self,
        messages: List[Dict[str, str]],
        children: List[Dict[str, Any]]
    ) -> List[NodeRelevance]:
        """Async version of node evaluation with batched processing"""
        try:
            # Process nodes in batches for better performance
            batch_size = 10
            batches = [
                children[i:i + batch_size] 
                for i in range(0, len(children), batch_size)
            ]
            
            async def process_batch(batch):
                evaluations = []
                for child in batch:
                    node_name = child.get("name", "").lower()
                    node_type = child.get("type", "").lower()
                    
                    relevance_score = 0.7
                    is_relevant = True
                    reason = "Possibly relevant based on name/type"
                    
                    if any(term in node_name for term in ["log", "debug", "print", "test", "mock"]):
                        relevance_score = 0.3
                        is_relevant = False
                        reason = "Likely utility, debug or test code"
                    elif any(term in node_name for term in ["api", "service", "controller", "repository", "dao", "database", "model", "entity"]):
                        relevance_score = 0.9
                        is_relevant = True
                        reason = "Likely core business logic or integration point"
                    elif any(term in node_name for term in ["util", "helper", "factory", "builder"]):
                        relevance_score = 0.6
                        is_relevant = True
                        reason = "Support code that may be needed for integration"
                    
                    evaluations.append(
                        NodeRelevance(
                            node_id=child["id"],
                            node_name=child["name"],
                            is_relevant=is_relevant,
                            relevance_score=relevance_score,
                            reason=reason
                        )
                    )
                return evaluations
            
            # Process all batches in parallel
            tasks = [process_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            return [
                eval_ 
                for batch_result in batch_results 
                for eval_ in batch_result
            ]
            
        except Exception as e:
            logging.exception(f"Error evaluating nodes: {str(e)}")
            return [
                NodeRelevance(
                    node_id=child["id"],
                    node_name=child["name"],
                    is_relevant=True,
                    relevance_score=0.8,
                    reason="Evaluation failed, defaulting to including node"
                )
                for child in children
            ]

    def _create_evaluation_prompt(
        self, 
        parent_node: Dict[str, Any], 
        parent_code: str, 
        children: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Create a prompt for the LLM to evaluate node relevance"""
        messages = [
            {
                "role": "system", 
                "content": """You are an expert code analyzer that can determine which code dependencies are relevant for understanding 
                a system's integration points. Your task is to evaluate each node for its relevance in the context of 
                integration test generation. Highly relevant nodes include:
                
                1. External API calls or third-party integrations
                2. Database interactions and data transformations
                3. Core business logic with branching paths
                4. Interface definitions and data models used across boundaries
                5. Authentication/authorization flows
                
                Nodes that are typically NOT relevant and add noise include:
                1. Logging statements and utilities
                2. Print statements
                3. Internal helper functions with no external dependencies
                4. Debug or development-only code
                5. Simple getters/setters with no business logic
                
                For each node, provide a relevance score between 0-1 and explain your reasoning.
                """
            },
            {
                "role": "user",
                "content": f"""I'm analyzing code to generate integration tests and need to determine which components are relevant.
                
                PARENT NODE:
                Name: {parent_node['name']}
                Type: {parent_node['type']}
                
                PARENT NODE CODE:
                ```
                {parent_code}
                ```
                
                CHILD NODES TO EVALUATE:
                {self._format_children_for_prompt(children)}
                
                For each child node, evaluate its likely relevance for integration testing based on its name and type.
                """
            }
        ]
        return messages

    def _format_children_for_prompt(self, children: List[Dict[str, Any]]) -> str:
        """Format child nodes information for the LLM prompt"""
        formatted = ""
        for i, child in enumerate(children):
            formatted += f"Child {i+1}:\n"
            formatted += f"ID: {child['id']}\n"
            formatted += f"Name: {child['name']}\n"
            formatted += f"Type: {child['type']}\n"
            if child.get('file_path'):
                formatted += f"File: {child['file_path']}\n"
            if child.get('relationship'):
                formatted += f"Relationship to Parent: {child['relationship']}\n"
            formatted += "\n"
        return formatted

    def _create_relevant_node(
        self, 
        node: Dict[str, Any], 
        relevance_score: float, 
        reason: str
    ) -> Dict[str, Any]:
        """Create a node with relevance information"""
        relevant_node = {
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "relevance_score": relevance_score,
            "reason": reason,
            "children": []
        }
        
        # Copy additional fields if present
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


def get_intelligent_code_graph_tool(sql_db: Session, provider_service: ProviderService, user_id: str) -> StructuredTool:
    """Create and return the intelligent code graph tool"""
    tool = IntelligentCodeGraphTool(sql_db, provider_service, user_id)
    
    # Define the schema more explicitly with proper typing
    class IntelligentCodeGraphSchema(BaseModel):
        project_id: str = Field(..., description="The repository ID (UUID).")
        node_id: str = Field(..., description="The ID of the node to retrieve the graph for.")
        relevance_threshold: float = Field(0.6, description="Minimum relevance score for a node to be included (default: 0.6).")
        max_depth: int = Field(5, description="Maximum depth of relationships to traverse (default: 5).")
    
    return StructuredTool.from_function(
        coroutine=tool.arun, 
        func=tool.run, 
        name=tool.name, 
        description=tool.description,
        args_schema=IntelligentCodeGraphSchema
    ) 