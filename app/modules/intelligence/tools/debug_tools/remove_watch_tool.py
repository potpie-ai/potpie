"""Remove a persistent watch expression for the current debug session."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import get_context_vars
from .watch_store import remove_watch


class RemoveWatchInput(BaseModel):
    expression: str = Field(description="Watch expression to remove")


def remove_watch_tool(input_data: RemoveWatchInput) -> str:
    user_id, conversation_id = get_context_vars()
    watches = remove_watch(user_id, conversation_id, input_data.expression)
    if watches:
        return f"Watch removed. Remaining watches ({len(watches)}):\n" + "\n".join(f"  - {w}" for w in watches)
    return "Watch removed. No active watches."


def create_remove_watch_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=remove_watch_tool,
        name="remove_watch",
        description="Remove a persistent watch expression from this debug session.",
        args_schema=RemoveWatchInput,
    )
