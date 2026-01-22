"""
Search Text Tool

Search for text patterns across files using LocalServer (grep-like functionality).
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchTextInput(BaseModel):
    query: str = Field(description="Text pattern to search for (supports regex if use_regex=True)")
    file_pattern: Optional[str] = Field(default=None, description="Glob pattern to filter files (e.g., '**/*.ts')")
    case_sensitive: bool = Field(default=False, description="Whether search is case-sensitive")
    use_regex: bool = Field(default=False, description="Whether query is a regex pattern")
    max_results: Optional[int] = Field(default=100, description="Maximum number of matches to return")
    use_bash: bool = Field(default=False, description="Use bash/grep command instead of file reading (faster for large codebases)")


def search_text_tool(input_data: SearchTextInput) -> str:
    """Search for text patterns across files using LocalServer (grep-like functionality).
    
    Can use either file reading (default) or bash/grep commands (faster for large codebases).
    """
    logger.info(f"🔍 [Tool Call] search_text_tool: Searching for '{input_data.query}' (use_bash={input_data.use_bash})")
    
    user_id, conversation_id = get_context_vars()
    
    # Use bash/grep if requested
    if input_data.use_bash:
        # Build grep command
        grep_flags = []
        if not input_data.case_sensitive:
            grep_flags.append("-i")  # Case-insensitive
        
        if input_data.use_regex:
            grep_flags.append("-E")  # Extended regex
        
        grep_flags.append("-r")  # Recursive
        grep_flags.append("-n")  # Show line numbers
        
        # Escape query for shell
        import shlex
        query = shlex.quote(input_data.query)
        
        # Build file pattern for grep
        file_pattern = input_data.file_pattern or "**/*"
        
        # Build command - use find + grep for better file pattern support
        if file_pattern and file_pattern != "**/*":
            # Convert glob pattern to find pattern
            # Simple conversion: **/*.py -> *.py
            find_pattern = file_pattern.replace("**/", "").replace("**", "*")
            # Escape for shell
            find_pattern = shlex.quote(find_pattern)
            command = f"find . -type f -name {find_pattern} -exec grep {' '.join(grep_flags)} {query} {{}} +"
        else:
            # Search all files
            command = f"grep {' '.join(grep_flags)} {query} ."
        
        result = route_to_local_server(
            "search_bash",
            {
                "command": command,
            },
            user_id=user_id,
            conversation_id=conversation_id,
        )
        
        if result:
            return result
    
    # Default: use file reading approach
    result = route_to_local_server(
        "search_text",
        {
            "query": input_data.query,
            "file_pattern": input_data.file_pattern,
            "case_sensitive": input_data.case_sensitive,
            "use_regex": input_data.use_regex,
            "max_results": input_data.max_results,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        return result
    
    return "❌ Search requires LocalServer connection. Please ensure tunnel is active."
