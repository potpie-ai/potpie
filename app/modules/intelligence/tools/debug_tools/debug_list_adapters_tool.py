"""List available debug adapters in the VS Code workspace.

Routes to the VS Code extension via the debug tunnel.
Extension endpoint: POST /api/debug/adapters
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import get_context_vars
from .debug_tunnel_utils import route_debug_command


class DebugListAdaptersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def debug_list_adapters_tool(_input_data: DebugListAdaptersInput) -> str:
    user_id, conversation_id = get_context_vars()
    result = route_debug_command("debug_list_adapters", {}, user_id, conversation_id)
    return result


def create_debug_list_adapters_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_list_adapters_tool,
        name="debug_list_adapters",
        description=(
            "List available debug adapters installed in VS Code (python/debugpy, node, go/delve). "
            "Returns language, availability flag, and extension id for each adapter. "
            "Use the result to pick the correct adapter type for debug_start."
        ),
        args_schema=DebugListAdaptersInput,
    )
