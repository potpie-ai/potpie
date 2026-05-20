"""Continue execution until next breakpoint."""

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command


class DebugContinueInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Debug session ID")


def debug_continue_tool(input_data: DebugContinueInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    return route_debug_command("debug_continue", payload, user_id, conversation_id)


def create_debug_continue_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_continue_tool,
        name="debug_continue",
        description="Resume execution (does not wait for next pause); call debug_snapshot(wait=true) later.",
        args_schema=DebugContinueInput,
    )
