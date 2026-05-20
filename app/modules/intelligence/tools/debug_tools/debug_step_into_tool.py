"""Step execution (step into)."""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    get_context_vars,
)

from .debug_tunnel_utils import route_debug_command
from .watch_store import merge_into_expressions


class DebugStepInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Debug session ID")
    expressions: Optional[List[str]] = Field(
        default=None,
        description="Extra expressions to evaluate in the post-step snapshot. Persistent watches are merged in automatically.",
    )


def debug_step_into_tool(input_data: DebugStepInput) -> str:
    user_id, conversation_id = get_context_vars()
    payload = input_data.model_dump(exclude_none=True)
    merged = merge_into_expressions(user_id, conversation_id, payload.get("expressions"))
    if merged:
        payload["expressions"] = merged
    return route_debug_command("debug_step_into", payload, user_id, conversation_id)


def create_debug_step_into_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_step_into_tool,
        name="debug_step_into",
        description="Step debugger execution; returns a fresh snapshot after the step completes.",
        args_schema=DebugStepInput,
    )
