"""
Search Symbols Tool

Search for symbols (functions, classes, variables, etc.) in a specific file using LocalServer.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchSymbolsInput(BaseModel):
    file_path: str = Field(description="Path to the file to search for symbols in")


def search_symbols_tool(input_data: SearchSymbolsInput) -> str:
    """Search for symbols (functions, classes, variables, etc.) in a specific file using LocalServer."""
    logger.info(f"🔍 [Tool Call] search_symbols_tool: Searching symbols in '{input_data.file_path}'")
    
    user_id, conversation_id = get_context_vars()
    
    result = route_to_local_server(
        "search_symbols",
        {
            "file_path": input_data.file_path,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
