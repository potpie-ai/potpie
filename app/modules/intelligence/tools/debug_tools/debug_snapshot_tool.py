"""Capture stack, locals, and optional watch expressions."""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugSnapshotInput(BaseModel):
    expressions: Optional[List[str]] = Field(
        default=None,
        description="Expressions to evaluate in the current frame (e.g. len(items), user.email)",
    )
    wait: bool = Field(
        default=True,
        description="If true, wait until paused at a breakpoint or completion timeout",
    )
    timeout: float = Field(
        default=30.0,
        description="Seconds to wait when wait=true before failing",
    )
    session_id: Optional[str] = Field(default=None, description="Debug session ID")


def debug_snapshot_tool(input_data: DebugSnapshotInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    return route_debug_command("debug_snapshot", payload, user_id, conversation_id)


def create_debug_snapshot_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_snapshot_tool,
        name="debug_snapshot",
        description=(
            "Primary inspection tool: returns paused location, call stack, locals, "
            "and expression evaluations. With wait=true, resumes configuration "
            "when needed so the program runs to breakpoints."
        ),
        args_schema=DebugSnapshotInput,
    )
