from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    GetCodeFromProbableNodeNameTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    GetCodeFromMultipleNodeIdsTool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    KnowledgeGraphQueryTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    GetNodesFromTags,
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
from app.modules.intelligence.tools.change_detection.change_detection import (
    ChangeDetectionTool,
)
from app.modules.intelligence.tools.tool_schema import ToolInfo


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
            "get_code_from_node_id": GetCodeFromNodeIdTool(
                self.db, self.user_id
            ),
            "get_code_from_multiple_node_ids": GetCodeFromMultipleNodeIdsTool(
                self.db, self.user_id
            ),
            "ask_knowledge_graph_queries": KnowledgeGraphQueryTool(
                self.db, self.user_id
            ),
            "get_nodes_from_tags": GetNodesFromTags(
                self.db, self.user_id
            ),
            "get_code_from_node_name": GetCodeFromNodeNameTool(
                self.db, self.user_id
            ),
            "get_code_graph_from_node_id": GetCodeGraphFromNodeIdTool(
                self.db, self.user_id
            ),
            "get_code_graph_from_node_name": GetCodeGraphFromNodeNameTool(
                self.db, self.user_id
            ),
            "change_detection": ChangeDetectionTool(
                self.db, self.user_id
            ),
        }

    async def run_tool(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_id)
        if not tool:
            raise ValueError(f"Invalid tool_id: {tool_id}")
        return await tool.run(**params)

    def list_tools(self) -> List[ToolInfo]:
        return [
            ToolInfo(
                id=tool_id, name=tool.__class__.__name__, description=tool.description
            )
            for tool_id, tool in self.tools.items()
        ]
