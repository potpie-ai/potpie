"""List active debugger sessions handled by the extension."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugListSessionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def debug_list_sessions_tool(_input_data: DebugListSessionsInput) -> str:
    user_id, conversation_id = get_context_vars()
    return route_debug_command("debug_list_sessions", {}, user_id, conversation_id)


def create_debug_list_sessions_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_list_sessions_tool,
        name="debug_list_sessions",
        description="List debug sessions for this workspace (read-only).",
        args_schema=DebugListSessionsInput,
    )
