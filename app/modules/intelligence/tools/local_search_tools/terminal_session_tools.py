"""
Terminal Session Management Tools

Tools for managing async terminal command sessions (getting output, sending signals, etc.).
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import (
    get_terminal_session_output,
    send_terminal_session_signal,
    get_context_vars,
)

logger = setup_logger(__name__)


class TerminalSessionOutputInput(BaseModel):
    session_id: str = Field(description="The session ID from async command execution")
    offset: Optional[int] = Field(
        default=0, description="Byte offset to read from (default: 0)"
    )


def terminal_session_output_tool(input_data: TerminalSessionOutputInput) -> str:
    """Get output from an async terminal session.

    Use this tool to poll for output from a long-running command that was started
    with `execute_terminal_command` in async mode.

    The tool returns incremental output from the specified offset, allowing you to
    stream output from long-running processes.

    **Example:**
    1. Start a command in async mode: `execute_terminal_command("npm run dev", mode="async")`
    2. Get session_id from the response
    3. Poll for output: `terminal_session_output(session_id="...", offset=0)`
    4. Use the returned offset for subsequent calls to get new output
    """
    logger.info(
        f"📥 [Tool Call] terminal_session_output: Getting output for session {input_data.session_id}"
    )

    user_id, conversation_id = get_context_vars()

    result = get_terminal_session_output(
        session_id=input_data.session_id,
        offset=input_data.offset or 0,
        user_id=user_id,
        conversation_id=conversation_id,
    )

    if result:
        output = result.get("output", "")
        error = result.get("error", "")
        offset = result.get("offset", input_data.offset or 0)
        is_complete = result.get("complete", False)

        formatted = f"📥 **Session Output** (session: `{input_data.session_id}`):\n\n"

        if output:
            # Limit output to 5000 chars
            output_preview = output[:5000]
            formatted += f"```\n{output_preview}\n```\n"
            if len(output) > 5000:
                formatted += f"\n... (output truncated, {len(output)} total characters)"

        if error:
            formatted += f"\n⚠️ **Error output:**\n```\n{error[:1000]}\n```"
            if len(error) > 1000:
                formatted += f"\n... (error truncated, {len(error)} total characters)"

        if is_complete:
            formatted += f"\n\n✅ **Process completed**"
        else:
            formatted += (
                f"\n\n📊 **Next offset:** {offset} (use this for subsequent calls)"
            )

        logger.info("✅ [terminal_session_output] Retrieved output successfully")
        return formatted

    logger.warning("⚠️ [terminal_session_output] Failed to get session output")
    return (
        f"❌ Failed to get output for session `{input_data.session_id}`.\n\n"
        "Possible reasons:\n"
        "- Session ID is invalid or expired\n"
        "- Tunnel connection is not active\n"
        "- Session was already closed"
    )


class TerminalSessionSignalInput(BaseModel):
    session_id: str = Field(description="The session ID to send signal to")
    signal: Optional[str] = Field(
        default="SIGINT",
        description="Signal to send (default: SIGINT). Common signals: SIGINT (Ctrl+C), SIGTERM (terminate), SIGKILL (force kill)",
    )


def terminal_session_signal_tool(input_data: TerminalSessionSignalInput) -> str:
    """Send a signal to a terminal session (e.g., SIGINT to stop a process).

    Use this tool to control long-running processes started in async mode.

    **Common signals:**
    - `SIGINT` (default) - Interrupt signal (equivalent to Ctrl+C)
    - `SIGTERM` - Termination signal (graceful shutdown)
    - `SIGKILL` - Force kill (cannot be caught or ignored)

    **Example:**
    1. Start a dev server: `execute_terminal_command("npm run dev", mode="async")`
    2. Get session_id
    3. When done, stop it: `terminal_session_signal(session_id="...", signal="SIGINT")`
    """
    logger.info(
        f"📤 [Tool Call] terminal_session_signal: Sending {input_data.signal} to session {input_data.session_id}"
    )

    user_id, conversation_id = get_context_vars()

    result = send_terminal_session_signal(
        session_id=input_data.session_id,
        signal=input_data.signal or "SIGINT",
        user_id=user_id,
        conversation_id=conversation_id,
    )

    if result:
        logger.info("✅ [terminal_session_signal] Signal sent successfully")
        return (
            f"✅ **Signal sent** (`{input_data.signal}` to session `{input_data.session_id}`)\n\n"
            "The process should respond to the signal. Use `terminal_session_output` to check the final state."
        )

    logger.warning("⚠️ [terminal_session_signal] Failed to send signal")
    return (
        f"❌ Failed to send signal `{input_data.signal}` to session `{input_data.session_id}`.\n\n"
        "Possible reasons:\n"
        "- Session ID is invalid or expired\n"
        "- Tunnel connection is not active\n"
        "- Session was already closed"
    )
