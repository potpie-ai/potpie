import asyncio
import logging
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project

class GetCodeGraphFromNodeIdTool:
    name = "get_code_graph_from_node_id"
    description = """Retrieves a code graph showing relationships between nodes starting from a specific node ID.
        :param project_id: string, the repository ID (UUID).
        :param node_id: string, the ID of the node to retrieve the graph for.
        :param max_depth: integer, optional, maximum depth of relationships to traverse (default: 3).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "node_id": "123e4567-e89b-12d3-a456-426614174000",
                "max_depth": 3
            }
    """

    def __init__(self, sql_db: Session):
        self.sql_db = sql_db
        cfg = config_provider.get_neo4j_config()
        self.neo4j_driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["username"], cfg["password"]))

    async def arun(self, project_id: str, node_id: str, max_depth: int = 3) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, node_id, max_depth)

    def run(self, project_id: str, node_id: str, max_depth: int = 3) -> Dict[str, Any]:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": f"Project '{project_id}' not found"}

        try:
            graph_data = self._get_graph_data(project_id, node_id, max_depth)
            if not graph_data:
                return {"error": f"No graph data found for node ID '{node_id}' in repo '{project_id}'"}

            return {"graph": {"name": project.repo_name, "repo_name": project.repo_name, "branch_name": project.branch_name, "root_node": graph_data}}
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def _get_graph_data(self, project_id: str, node_id: str, max_depth: int) -> Optional[Dict[str, Any]]:
        # Query to get the root node
        node_query = """
MATCH (node:NODE {node_id: $node_id, repoId: $project_id})
RETURN {
    id: node.node_id, 
    name: node.name, 
    type: head(labels(node)),
    file_path: node.file_path, 
    start_line: node.start_line, 
    end_line: node.end_line, 
    children: []
} AS node_data
"""
        # Optimized relationship query with better filtering
        relationship_query = """
MATCH (n:NODE {node_id: $parent_id, repoId: $project_id})-[r]->(child:NODE {repoId: $project_id})
WHERE type(r) IN ['CONTAINS','CALLS','REFERENCES','FUNCTION_DEFINITION','IMPORTS','INSTANTIATES','CLASS_DEFINITION'] 
  AND type(r) <> 'IS_LEAF'
  AND (child:CLASS OR child:FUNCTION OR child:METHOD OR child:INTERFACE OR child:FILE)
RETURN n.node_id AS parent_id, {
    id: child.node_id, 
    name: child.name, 
    type: head(labels(child)),
    file_path: child.file_path, 
    start_line: child.start_line, 
    end_line: child.end_line, 
    relationship: type(r)
} AS child_data
ORDER BY child.start_line
"""
        try:
            with self.neo4j_driver.session() as session:
                # Get the root node
                record = session.run(node_query, node_id=node_id, project_id=project_id).single()
                if not record:
                    return None
                root = record["node_data"]
                
                # Track visited nodes to avoid cycles
                visited = set([root["id"]])
                
                # Breadth-first traversal
                current_level = [root["id"]]
                for level in range(max_depth):
                    if not current_level:
                        break
                        
                    next_level = []
                    for parent_id in current_level:
                        # Fetch and process relationships for this parent
                        for rec in session.run(relationship_query, parent_id=parent_id, project_id=project_id):
                            child = rec["child_data"]
                            child_id = child["id"]
                            
                            # Skip if we've already processed this node
                            if child_id in visited:
                                continue
                                
                            # Mark as visited
                            visited.add(child_id)
                            
                            # Attach to the correct parent node
                            self._attach_child(root, rec["parent_id"], child)
                            
                            # Add to next level for further processing
                            next_level.append(child_id)
                    
                    # Update for next iteration
                    current_level = next_level

                return root
        except Exception as e:
            logging.error(f"Error in _get_graph_data: {str(e)}")
            return None

    def _attach_child(self, node: Dict[str, Any], parent_id: str, child: Dict[str, Any]):
        # Initialize children list if missing
        if "children" not in node:
            node["children"] = []
            
        if node["id"] == parent_id:
            node["children"].append(child)
        else:
            for c in node["children"]:
                self._attach_child(c, parent_id, child)

def get_code_graph_from_node_id_tool(sql_db: Session) -> StructuredTool:
    tool = GetCodeGraphFromNodeIdTool(sql_db)
    return StructuredTool.from_function(coroutine=tool.arun, func=tool.run, name=tool.name, description=tool.description)