"""
Execute Terminal Command Tool

Execute shell commands on the user's local machine via the VS Code extension
(Socket.IO workspace connection). Supports both sync and async execution modes.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import (
    route_terminal_command,
    format_terminal_result,
    get_context_vars,
)

logger = setup_logger(__name__)


class ExecuteTerminalCommandInput(BaseModel):
    command: str = Field(
        description="The shell command to execute (e.g., 'npm test', 'git status', 'python script.py')"
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory relative to workspace root (optional)",
    )
    timeout: Optional[int] = Field(
        default=30000, description="Timeout in milliseconds (default: 30000)"
    )
    mode: Optional[str] = Field(
        default="sync",
        description="Execution mode: 'sync' (default) for immediate results, 'async' for long-running commands",
    )


def execute_terminal_command_tool(input_data: ExecuteTerminalCommandInput) -> str:
    """Execute a shell command on the user's local machine via the VS Code extension (Socket.IO).

    This tool executes commands directly on the user's local machine through the VS Code extension.
    Commands run within the workspace directory with security restrictions.

    **Sync Mode (default):**
    - Returns immediate results
    - Best for short commands (tests, builds, git status, etc.)
    - Timeout applies to the entire command execution

    **Async Mode:**
    - Returns a session_id for long-running commands
    - Use terminal_session_output tool to poll for output
    - Use terminal_session_signal tool to stop the process

    **Security:**
    - Commands are validated - dangerous commands (rm, sudo, etc.) are blocked by default
    - Commands run within the VS Code workspace scope
    - Write operations may be restricted

    **Examples:**
    - `npm test` - Run tests
    - `git status` - Check git status
    - `python script.py` - Run a Python script
    - `npm run dev` (async mode) - Start a dev server
    """
    logger.info(
        f"🔧 [Tool Call] execute_terminal_command: Executing '{input_data.command}' (mode: {input_data.mode})"
    )

    user_id, conversation_id = get_context_vars()

    # Route to LocalServer via workspace socket (Socket.IO)
    result, error_type = route_terminal_command(
        command=input_data.command,
        working_directory=input_data.working_directory,
        timeout=input_data.timeout or 30000,
        mode=input_data.mode or "sync",
        user_id=user_id,
        conversation_id=conversation_id,
    )

    if result:
        logger.info("✅ [execute_terminal_command] Executed via workspace socket (Socket.IO)")

        # For async mode, return session info
        if input_data.mode == "async" and result.get("session_id"):
            session_id = result.get("session_id")
            return (
                f"🔄 **Command started in async mode** (`{input_data.command}`):\n\n"
                f"**Session ID:** `{session_id}`\n\n"
                f"Use `terminal_session_output` tool with session_id `{session_id}` to get output.\n"
                f"Use `terminal_session_signal` tool to stop the process if needed."
            )

        # Format and return result
        return format_terminal_result(result)

    # Handle different error types with specific messages (lazy import to avoid circular import with code_changes_manager)
    from app.modules.tunnel.tunnel_service import get_tunnel_service
    from app.modules.intelligence.tools.code_changes_manager import (
        _get_repository,
        _get_branch,
    )

    tunnel_service = get_tunnel_service()
    tunnel_url = (
        tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            repository=_get_repository(),
            branch=_get_branch(),
        )
        if user_id
        else None
    )

    if error_type == "tunnel_unreachable" or (
        tunnel_url and error_type in ["timeout", "connection_error"]
    ):
        # Workspace socket was registered but request failed
        logger.warning(
            f"⚠️ [execute_terminal_command] Workspace socket registered ({tunnel_url}) but not reachable (error: {error_type})"
        )
        return (
            f"❌ **Workspace socket connection error**\n\n"
            f"The extension is registered (`{tunnel_url}`) but the request failed.\n\n"
            f"**Possible causes:**\n"
            f"- The extension Socket.IO connection was interrupted or disconnected\n"
            f"- Network connectivity issues\n"
            f"- Temporary connection timeout\n\n"
            f"**To fix:**\n"
            f"1. Ensure the VS Code extension is running and connected\n"
            f"2. Check the extension shows a connected status\n"
            f"3. Try the command again - this may be a transient connection issue\n"
            f"4. The extension should automatically reconnect if needed\n"
            f"5. If the issue persists, check the VS Code extension logs\n\n"
            f"**Note:** The workspace socket may be temporarily unavailable. The extension will automatically reconnect."
        )
    elif error_type == "tunnel_expired":
        # Workspace registration was expired and cleaned up
        logger.warning("⚠️ [execute_terminal_command] Workspace socket registration expired and was cleaned up")
        return (
            f"❌ **Workspace socket registration expired**\n\n"
            f"The extension registration has expired and has been cleaned up.\n\n"
            f"**To fix:**\n"
            f"1. The VS Code extension should automatically re-register\n"
            f"2. Try the command again in a few seconds\n"
            f"3. If the issue persists, check the VS Code extension logs\n"
            f"4. Ensure the extension is running and connected"
        )
    elif error_type == "no_tunnel" or error_type == "no_user_id":
        # No workspace socket available: no user_id, or no workspace_id (repo not in context), or socket not registered
        logger.warning(
            f"⚠️ [execute_terminal_command] Workspace socket not available (error: {error_type})"
        )
        return (
            "❌ Terminal command requires the VS Code extension (Socket.IO workspace connection) for local execution.\n\n"
            "**Local execution (required):**\n"
            "- Runs on your machine via the VS Code extension over a Socket.IO connection\n"
            "- Requires the conversation to have a linked project/repo and the extension to be connected\n"
            "- Commands are validated for security\n\n"
            "**To fix:**\n"
            "1. Ensure the VS Code extension is installed and connected (Socket.IO to this backend)\n"
            "2. Start the conversation from a project/workspace so the backend knows which repo (workspace_id) to use\n"
            "3. If the extension is connected but this still appears, check that the extension has registered the workspace (register_workspace after auth)\n"
            "4. Check the extension logs for connection/registration errors\n"
            "5. Try reloading the VS Code window"
        )
    else:
        # Unknown error
        logger.warning(f"⚠️ [execute_terminal_command] Unknown error: {error_type}")
        return (
            f"❌ **Terminal command failed**\n\n"
            f"An error occurred while executing the command via the workspace socket.\n\n"
            f"**Error type:** {error_type}\n\n"
            f"**To fix:**\n"
            f"1. Try the command again - this may be a transient issue\n"
            f"2. Check the VS Code extension logs for more details\n"
            f"3. Ensure the extension is connected and the workspace is registered"
        )
