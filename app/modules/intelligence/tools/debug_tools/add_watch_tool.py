"""Add a persistent watch expression for the current debug session."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import get_context_vars
from .watch_store import add_watch


class AddWatchInput(BaseModel):
    expression: str = Field(
        description="Expression to evaluate in every subsequent snapshot (e.g. 'len(items)', 'user.email', 'error.constructor.name')"
    )


def add_watch_tool(input_data: AddWatchInput) -> str:
    user_id, conversation_id = get_context_vars()
    watches = add_watch(user_id, conversation_id, input_data.expression)
    return f"Watch added. Active watches ({len(watches)}):\n" + "\n".join(f"  - {w}" for w in watches)


def create_add_watch_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=add_watch_tool,
        name="add_watch",
        description=(
            "Register a persistent watch expression for this debug session. "
            "The expression will be automatically evaluated in every debug_snapshot, "
            "debug_step_*, and debug_select_frame call."
        ),
        args_schema=AddWatchInput,
    )
