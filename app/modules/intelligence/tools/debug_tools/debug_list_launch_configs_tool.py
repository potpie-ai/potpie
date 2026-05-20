"""List available VS Code launch.json debug configurations.

Routes to the VS Code extension via the debug tunnel.
Extension endpoint: POST /api/debug/launch-configs
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import get_context_vars
from .debug_tunnel_utils import route_debug_command


class DebugListLaunchConfigsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def debug_list_launch_configs_tool(_input_data: DebugListLaunchConfigsInput) -> str:
    user_id, conversation_id = get_context_vars()
    result = route_debug_command("debug_list_launch_configs", {}, user_id, conversation_id)
    return result


def create_debug_list_launch_configs_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=debug_list_launch_configs_tool,
        name="debug_list_launch_configs",
        description=(
            "List all VS Code launch.json debug configurations available in the workspace. "
            "Returns config names, types (python/node/go), and entry points. "
            "Use this to choose the right config before calling debug_start."
        ),
        args_schema=DebugListLaunchConfigsInput,
    )
