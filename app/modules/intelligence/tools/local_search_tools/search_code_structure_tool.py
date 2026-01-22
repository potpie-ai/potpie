"""
Search Code Structure Tool

Search for code structure (classes, functions, etc.) using LocalServer.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchCodeStructureInput(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Path to file to search in (if searching in specific file)")
    query: Optional[str] = Field(default=None, description="Symbol name to search for (if searching across workspace)")
    kind: Optional[str] = Field(
        default=None,
        description="Filter by symbol kind: 'class', 'function', 'method', 'variable', 'interface', etc."
    )


def search_code_structure_tool(input_data: SearchCodeStructureInput) -> str:
    """Search for code structure (classes, functions, etc.) using LocalServer."""
    logger.info(f"🔍 [Tool Call] search_code_structure_tool: file_path={input_data.file_path}, query={input_data.query}, kind={input_data.kind}")
    
    user_id, conversation_id = get_context_vars()
    
    result = route_to_local_server(
        "search_code_structure",
        {
            "file_path": input_data.file_path,
            "query": input_data.query,
            "kind": input_data.kind,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
