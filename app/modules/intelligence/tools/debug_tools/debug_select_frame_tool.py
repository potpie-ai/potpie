"""Select a stack frame index for inspection (locals scoped to that frame)."""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command
from .watch_store import merge_into_expressions


class DebugSelectFrameInput(BaseModel):
    frame_index: int = Field(ge=0, description="0-based index into the call stack (0 = top)")
    session_id: Optional[str] = Field(default=None, description="Debug session ID")
    expressions: Optional[List[str]] = Field(
        default=None,
        description="Extra expressions to evaluate in the selected frame. Persistent watches are merged in automatically.",
    )


def debug_select_frame_tool(input_data: DebugSelectFrameInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    merged = merge_into_expressions(user_id, conversation_id, payload.get("expressions"))
    if merged:
        payload["expressions"] = merged
    return route_debug_command(
        "debug_select_frame", payload, user_id, conversation_id
    )


def create_debug_select_frame_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_select_frame_tool,
        name="debug_select_frame",
        description="Inspect locals and expressions for a specific stack frame.",
        args_schema=DebugSelectFrameInput,
    )
