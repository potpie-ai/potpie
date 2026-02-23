"""
Search Workspace Symbols Tool

Search for symbols across the entire workspace using LocalServer.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchWorkspaceSymbolsInput(BaseModel):
    query: str = Field(description="Symbol name or pattern to search for")
    max_results: Optional[int] = Field(default=100, description="Maximum number of results to return")


def search_workspace_symbols_tool(input_data: SearchWorkspaceSymbolsInput) -> str:
    """Search for symbols across the entire workspace using LocalServer."""
    logger.info(f"🔍 [Tool Call] search_workspace_symbols_tool: Searching for '{input_data.query}'")
    
    user_id, conversation_id = get_context_vars()
    
    result = route_to_local_server(
        "search_workspace_symbols",
        {
            "query": input_data.query,
            "max_results": input_data.max_results,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
