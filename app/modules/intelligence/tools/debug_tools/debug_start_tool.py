"""Launch or attach a debug session via the VS Code extension."""

from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugStartInput(BaseModel):
    program: str = Field(description="Entry script path relative to workspace root (e.g. src/app.py)")
    language: str = Field(default="python", description="Language/runtime (initial support: python)")
    mode: str = Field(default="launch", description="'launch' or 'attach'")
    port: Optional[int] = Field(default=None, description="Port when mode is attach")
    args: Dict[str, Any] = Field(default_factory=dict, description="Extra launch arguments for the adapter")


def debug_start_tool(input_data: DebugStartInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    return route_debug_command("debug_start", payload, user_id, conversation_id)


def create_debug_start_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_start_tool,
        name="debug_start",
        description=(
            "Start a debugger session under the VS Code extension (local mode). "
            "Sets up an adapter session; use debug_set_breakpoints before debug_snapshot(wait=True) "
            "so breakpoints apply before execution begins."
        ),
        args_schema=DebugStartInput,
    )
