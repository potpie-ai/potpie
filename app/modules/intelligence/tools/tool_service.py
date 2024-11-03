from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.modules.intelligence.tools.change_detection.change_detection_tool import (
    ChangeDetectionTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    GetCodeFileStructureTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_from_node_name_tool import (
    GetCodeFromNodeNameTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_name_tool import (
    GetCodeGraphFromNodeNameTool,
)
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    GetNodeNeighboursFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    KnowledgeGraphQueryTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    GetCodeFromMultipleNodeIdsTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    GetCodeFromProbableNodeNameTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    GetNodesFromTags,
)
from app.modules.intelligence.tools.tool_schema import ToolInfo, ToolParameter


class ToolService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Any]:
        return {
            "get_code_from_probable_node_name": GetCodeFromProbableNodeNameTool(
                self.db, self.user_id
            ),
            "get_code_from_node_id": GetCodeFromNodeIdTool(self.db, self.user_id),
            "get_code_from_multiple_node_ids": GetCodeFromMultipleNodeIdsTool(
                self.db, self.user_id
            ),
            "ask_knowledge_graph_queries": KnowledgeGraphQueryTool(
                self.db, self.user_id
            ),
            "get_nodes_from_tags": GetNodesFromTags(self.db, self.user_id),
            "get_code_from_node_name": GetCodeFromNodeNameTool(self.db, self.user_id),
            "get_code_graph_from_node_id": GetCodeGraphFromNodeIdTool(self.db),
            "get_code_graph_from_node_name": GetCodeGraphFromNodeNameTool(self.db),
            "change_detection": ChangeDetectionTool(self.db, self.user_id),
            "get_code_file_structure": GetCodeFileStructureTool(self.db),
            "get_node_neighbours_from_node_id": GetNodeNeighboursFromNodeIdTool(
                self.db
            ),
        }

    async def run_tool(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_id)
        if not tool:
            raise ValueError(f"Invalid tool_id: {tool_id}")

        # If the tool has an arun method, use it
        if hasattr(tool, "arun"):
            return await tool.arun(**params)
        else:
            raise ValueError(f"Tool {tool.__class__.__name__} has no arun method")

    def list_tools(self) -> List[ToolInfo]:
        return [
            ToolInfo(
                id=tool_id,
                name=tool.__class__.__name__,
                description=tool.description,
                # parameters=self._get_tool_parameters(tool),
            )
            for tool_id, tool in self.tools.items()
        ]