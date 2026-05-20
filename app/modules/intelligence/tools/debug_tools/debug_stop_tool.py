"""Stop an active debug session."""

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugStopInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID from debug_start; omit if single session")


def debug_stop_tool(input_data: DebugStopInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    return route_debug_command("debug_stop", payload, user_id, conversation_id)


def create_debug_stop_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_stop_tool,
        name="debug_stop",
        description="Terminate the debug session and disconnect from the adapter.",
        args_schema=DebugStopInput,
    )
