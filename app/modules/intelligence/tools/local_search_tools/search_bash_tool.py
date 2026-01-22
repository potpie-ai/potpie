"""
Search Bash Tool

Execute bash commands (grep, find, etc.) locally via LocalServer.
Falls back to cloud bash_command tool if LocalServer is not available.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchBashInput(BaseModel):
    command: str = Field(description="Bash command to execute (read-only commands only: grep, find, awk, etc.)")
    working_directory: Optional[str] = Field(default=None, description="Optional subdirectory to run command in")
    timeout: Optional[int] = Field(default=30000, description="Command timeout in milliseconds (default: 30000)")
    project_id: Optional[str] = Field(default=None, description="Project ID for cloud fallback (optional, will try to get from context)")


def search_bash_tool(input_data: SearchBashInput) -> str:
    """Execute bash commands locally via LocalServer (grep, find, awk, etc.).
    
    This tool tries to run commands locally first (via LocalServer tunnel).
    If LocalServer is not available, it falls back to cloud execution (bash_command tool).
    
    Local execution (preferred):
    - Runs directly in your VS Code workspace
    - Faster, uses your local environment
    - Requires tunnel connection
    
    Cloud execution (fallback):
    - Runs in gVisor sandbox on cloud worktree
    - Requires project_id and repo manager enabled
    - Slower but works without tunnel
    
    Allowed commands: grep, find, awk, sed, cat, head, tail, ls, wc, sort, uniq, etc.
    Blocked: rm, mv, cp, chmod, git, sudo, and any write operations.
    """
    logger.info(f"🔍 [Tool Call] search_bash_tool: Executing '{input_data.command}'")
    
    user_id, conversation_id = get_context_vars()
    
    # Try LocalServer first (local execution)
    result = route_to_local_server(
        "search_bash",
        {
            "command": input_data.command,
            "working_directory": input_data.working_directory,
            "timeout": input_data.timeout,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        logger.info("✅ [search_bash_tool] Executed locally via LocalServer")
        return result
    
    # Fallback to cloud bash_command tool
    logger.info("⚠️ [search_bash_tool] LocalServer not available, falling back to cloud bash_command")
    
    # Fallback to cloud bash_command tool (requires project_id)
    if not input_data.project_id:
        return (
            "❌ Bash command requires LocalServer connection (tunnel) for local execution.\n\n"
            "**Local execution (preferred):**\n"
            "- Runs directly in your VS Code workspace\n"
            "- Faster, uses your local environment\n"
            "- Requires tunnel connection\n\n"
            "**Cloud execution (fallback):**\n"
            "- Use `bash_command` tool instead with project_id parameter\n"
            "- Runs in gVisor sandbox on cloud worktree\n"
            "- Requires project_id and repo manager enabled\n\n"
            "Please ensure tunnel is active for local execution, or use `bash_command` tool for cloud execution."
        )
    
    # Use cloud bash_command tool
    try:
        from app.modules.projects.projects_service import ProjectService
        from app.core.database import get_db
        from app.modules.intelligence.tools.code_query_tools.bash_command_tool import BashCommandTool
        
        # Get database session
        db = next(get_db())
        
        # Create bash command tool instance
        bash_tool = BashCommandTool(db, user_id)
        
        # Execute command
        result = bash_tool._run(
            project_id=input_data.project_id,
            command=input_data.command,
            working_directory=input_data.working_directory,
        )
        
        if result.get("success"):
            output = result.get("output", "")
            error = result.get("error", "")
            exit_code = result.get("exit_code", 0)
            
            formatted = f"📋 **Bash command result** (cloud execution, `{input_data.command}`):\n\n"
            if output:
                formatted += f"```\n{output[:5000]}\n```\n"
                if len(output) > 5000:
                    formatted += f"\n... (output truncated, {len(output)} total characters)"
            if error:
                formatted += f"\n⚠️ **Error output:**\n```\n{error[:1000]}\n```"
            if exit_code != 0:
                formatted += f"\n⚠️ Exit code: {exit_code}"
            
            logger.info("✅ [search_bash_tool] Executed in cloud (fallback)")
            return formatted
        else:
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"❌ [search_bash_tool] Cloud execution failed: {error_msg}")
            return f"❌ Bash command failed (cloud execution): {error_msg}"
    
    except Exception as e:
        logger.exception(f"Error in cloud bash_command fallback: {e}")
        return (
            f"❌ Failed to execute bash command in cloud (fallback): {str(e)}\n\n"
            "Please ensure:\n"
            "1. Tunnel is active for local execution, OR\n"
            "2. project_id is valid and repo manager is enabled for cloud execution."
        )
