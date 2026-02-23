"""
Search Definitions Tool

Find definition of a symbol using LocalServer.
"""

from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchDefinitionsInput(BaseModel):
    file_path: str = Field(description="Path to the file containing the symbol")
    line: int = Field(description="Line number where the symbol appears (1-indexed)")
    character: int = Field(description="Character position in the line (1-indexed)")


def search_definitions_tool(input_data: SearchDefinitionsInput) -> str:
    """Find definition of a symbol using LocalServer."""
    logger.info(f"🔍 [Tool Call] search_definitions_tool: Finding definition at {input_data.file_path}:{input_data.line}:{input_data.character}")
    
    user_id, conversation_id = get_context_vars()
    
    result = route_to_local_server(
        "search_definitions",
        {
            "file_path": input_data.file_path,
            "line": input_data.line,
            "character": input_data.character,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
