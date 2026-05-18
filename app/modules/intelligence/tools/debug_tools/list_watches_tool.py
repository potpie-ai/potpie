"""List all persistent watch expressions for the current debug session."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import get_context_vars
from .watch_store import list_watches


class ListWatchesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def list_watches_tool(_input_data: ListWatchesInput) -> str:
    user_id, conversation_id = get_context_vars()
    watches = list_watches(user_id, conversation_id)
    if not watches:
        return "No active watches. Use `add_watch` to register expressions."
    return f"Active watches ({len(watches)}):\n" + "\n".join(f"  - {w}" for w in watches)


def create_list_watches_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=list_watches_tool,
        name="list_watches",
        description="List all persistent watch expressions registered for this debug session.",
        args_schema=ListWatchesInput,
    )
