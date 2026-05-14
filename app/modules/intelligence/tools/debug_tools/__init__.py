"""Debugger tools routed through the VS Code extension tunnel (local mode)."""

from typing import List

from langchain_core.tools import StructuredTool

from .debug_continue_tool import create_debug_continue_tool
from .debug_list_sessions_tool import create_debug_list_sessions_tool
from .debug_select_frame_tool import create_debug_select_frame_tool
from .debug_set_breakpoints_tool import create_debug_set_breakpoints_tool
from .debug_snapshot_tool import create_debug_snapshot_tool
from .debug_start_tool import create_debug_start_tool
from .debug_step_into_tool import create_debug_step_into_tool
from .debug_step_out_tool import create_debug_step_out_tool
from .debug_step_over_tool import create_debug_step_over_tool
from .debug_stop_tool import create_debug_stop_tool


def create_debug_tools() -> List[StructuredTool]:
    """Return all debugger StructuredTools for ToolService registration."""
    return [
        create_debug_start_tool(),
        create_debug_stop_tool(),
        create_debug_set_breakpoints_tool(),
        create_debug_snapshot_tool(),
        create_debug_step_into_tool(),
        create_debug_step_out_tool(),
        create_debug_step_over_tool(),
        create_debug_continue_tool(),
        create_debug_select_frame_tool(),
        create_debug_list_sessions_tool(),
    ]


__all__ = ["create_debug_tools"]
