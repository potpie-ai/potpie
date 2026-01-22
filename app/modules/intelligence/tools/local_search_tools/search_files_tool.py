"""
Search Files Tool

Search for files in workspace using LocalServer.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchFilesInput(BaseModel):
    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.ts', 'src/**/*.py')")
    exclude: Optional[str] = Field(default=None, description="Glob pattern to exclude files")
    max_results: Optional[int] = Field(default=100, description="Maximum number of results to return")


def search_files_tool(input_data: SearchFilesInput) -> str:
    """Search for files in workspace using LocalServer."""
    logger.info(f"🔍 [Tool Call] search_files_tool: Searching files with pattern '{input_data.pattern}'")
    
    user_id, conversation_id = get_context_vars()
    
    result = route_to_local_server(
        "search_files",
        {
            "pattern": input_data.pattern,
            "exclude": input_data.exclude,
            "max_results": input_data.max_results,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
