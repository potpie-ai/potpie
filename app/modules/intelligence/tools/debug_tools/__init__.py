"""Debugger tools routed through the VS Code extension tunnel (local mode)."""

from typing import List

from langchain_core.tools import StructuredTool

from .add_watch_tool import create_add_watch_tool
from .build_debug_context_tool import create_build_debug_context_tool
from .debug_continue_tool import create_debug_continue_tool
from .debug_list_adapters_tool import create_debug_list_adapters_tool
from .debug_list_launch_configs_tool import create_debug_list_launch_configs_tool
from .debug_list_sessions_tool import create_debug_list_sessions_tool
from .debug_select_frame_tool import create_debug_select_frame_tool
from .debug_set_breakpoints_tool import create_debug_set_breakpoints_tool
from .debug_snapshot_tool import create_debug_snapshot_tool
from .debug_start_tool import create_debug_start_tool
from .debug_step_into_tool import create_debug_step_into_tool
from .debug_step_out_tool import create_debug_step_out_tool
from .debug_step_over_tool import create_debug_step_over_tool
from .debug_stop_tool import create_debug_stop_tool
from .find_related_tests_tool import create_find_related_tests_tool
from .list_watches_tool import create_list_watches_tool
from .parse_debug_signal_tool import create_parse_debug_signal_tool
from .remove_watch_tool import create_remove_watch_tool


def create_debug_tools() -> List[StructuredTool]:
    """Return all debugger StructuredTools for ToolService registration."""
    return [
        # DAP / extension tunnel tools
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
        create_debug_list_launch_configs_tool(),
        create_debug_list_adapters_tool(),
        # Backend-only debug workflow tools
        create_parse_debug_signal_tool(),
        create_build_debug_context_tool(),
        create_find_related_tests_tool(),
        create_add_watch_tool(),
        create_remove_watch_tool(),
        create_list_watches_tool(),
    ]


__all__ = ["create_debug_tools"]
