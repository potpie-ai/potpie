"""Set breakpoints in a source file (workspace-relative path)."""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugSetBreakpointsInput(BaseModel):
    file: str = Field(description="Source file path relative to workspace root")
    lines: List[int] = Field(description="1-based lines where breakpoints should be set")
    condition: Optional[str] = Field(default=None, description="Optional breakpoint condition expression")


def debug_set_breakpoints_tool(input_data: DebugSetBreakpointsInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    return route_debug_command(
        "debug_set_breakpoints", payload, user_id, conversation_id
    )


def create_debug_set_breakpoints_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_set_breakpoints_tool,
        name="debug_set_breakpoints",
        description=(
            "Set breakpoints before starting execution. Prefer calling after debug_start "
            "and before the first debug_snapshot(wait=True)."
        ),
        args_schema=DebugSetBreakpointsInput,
    )
