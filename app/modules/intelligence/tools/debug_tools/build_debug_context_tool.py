"""Build a structured debug context packet from a parsed debug signal.

Orchestrates several existing tools (code lookup, git, related tests, launch configs,
adapters) and returns a consolidated markdown packet for the agent.

Each sub-tool failure is caught and noted gracefully — the packet is always returned
even if individual sections are empty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class BuildDebugContextInput(BaseModel):
    stack_frames: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Extracted stack frames from parse_debug_signal: list of {file, line, symbol}",
    )
    error_signature: Optional[str] = Field(
        default=None,
        description="Top-level error class or code (e.g. 'PaymentTimeoutError')",
    )
    signal_type: Optional[str] = Field(
        default=None,
        description="signal_type from parse_debug_signal (pasted_log / stack_trace / failed_test / natural_language)",
    )
    test_path: Optional[str] = Field(
        default=None,
        description="Failing test file path from parse_debug_signal (for failed_test signals)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(label: str, fn, *args, **kwargs) -> Optional[str]:
    try:
        result = fn(*args, **kwargs)
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        logger.debug(f"build_debug_context: [{label}] failed: {e}")
        return None


def _section(title: str, body: Optional[str]) -> str:
    if not body or not body.strip():
        return f"### {title}\n_Not available._\n"
    return f"### {title}\n{body.strip()}\n"


# ---------------------------------------------------------------------------
# Sub-tool wrappers
# ---------------------------------------------------------------------------

def _fetch_source_locations(frames: List[Dict[str, Any]]) -> str:
    """Fetch code snippets for each stack frame and produce a source-locations list."""
    if not frames:
        return ""
    try:
        from app.modules.intelligence.tools.code_tools.get_code_from_probable_node_name_tool import (
            GetCodeFromProbableNodeNameInput,
            get_code_from_probable_node_name_tool,
        )
    except ImportError:
        return ""

    lines = []
    for frame in frames[:6]:  # limit to top 6 frames
        sym = frame.get("symbol") or ""
        file_ = frame.get("file") or ""
        line_ = frame.get("line") or "?"
        confidence = "high" if sym and sym != "?" else "medium"
        lines.append(f"- **{file_}:{line_}** in `{sym}` — confidence: {confidence}")

        if sym and sym not in ("?", "<module>", "<anonymous>"):
            result = _safe_call(
                f"node_name/{sym}",
                lambda s=sym: get_code_from_probable_node_name_tool(
                    GetCodeFromProbableNodeNameInput(node_name=s)
                ),
            )
            if result and len(result) < 3000:
                lines.append(f"  ```\n  {result[:1500]}\n  ```")

    return "\n".join(lines)


def _fetch_related_tests(frames: List[Dict[str, Any]], test_path: Optional[str]) -> str:
    """Find related tests for the top stack frame."""
    if test_path:
        return f"- `{test_path}` (from signal)"

    if not frames:
        return ""

    try:
        from app.modules.intelligence.tools.debug_tools.find_related_tests_tool import (
            FindRelatedTestsInput,
            find_related_tests,
        )

        top = frames[0]
        result = _safe_call(
            "find_related_tests",
            find_related_tests,
            FindRelatedTestsInput(
                file_path=top.get("file"),
                symbol=top.get("symbol") if top.get("symbol") != "?" else None,
            ),
        )
        return result or ""
    except Exception:
        return ""


def _fetch_recent_git_changes(frames: List[Dict[str, Any]]) -> str:
    """Run git log + diff around suspect files."""
    if not frames:
        return ""
    try:
        from app.modules.intelligence.tools.sandbox.tool_functions import sandbox_git_fn

        suspect_files = [f["file"] for f in frames[:3] if f.get("file")]

        log_result = _safe_call(
            "git_log",
            sandbox_git_fn,
            command="log",
        )

        diff_result = None
        if suspect_files:
            diff_result = _safe_call(
                "git_diff",
                sandbox_git_fn,
                command="diff",
                paths=suspect_files,
            )

        parts = []
        if log_result:
            parts.append(f"**Recent commits:**\n{log_result[:1500]}")
        if diff_result:
            parts.append(f"**Diff for suspect files:**\n{diff_result[:2000]}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.debug(f"build_debug_context: git section failed: {e}")
        return ""


def _fetch_launch_configs() -> str:
    """List available VS Code launch.json configurations."""
    try:
        from app.modules.intelligence.tools.debug_tools.debug_list_launch_configs_tool import (
            DebugListLaunchConfigsInput,
            debug_list_launch_configs_tool,
        )

        result = _safe_call(
            "launch_configs",
            debug_list_launch_configs_tool,
            DebugListLaunchConfigsInput(),
        )
        return result or "_debug_list_launch_configs not yet available — read .vscode/launch.json manually._"
    except Exception:
        return "_debug_list_launch_configs not yet available — read .vscode/launch.json manually._"


def _fetch_adapters() -> str:
    """List available debug adapters."""
    try:
        from app.modules.intelligence.tools.debug_tools.debug_list_adapters_tool import (
            DebugListAdaptersInput,
            debug_list_adapters_tool,
        )

        result = _safe_call(
            "adapters",
            debug_list_adapters_tool,
            DebugListAdaptersInput(),
        )
        return result or "_debug_list_adapters not yet available._"
    except Exception:
        return "_debug_list_adapters not yet available._"


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

def build_debug_context(input_data: BuildDebugContextInput) -> str:
    frames = input_data.stack_frames or []
    error_sig = input_data.error_signature or ""
    signal_type = input_data.signal_type or "unknown"
    test_path = input_data.test_path

    sections: List[str] = [
        f"## Debug Context Packet\n\n"
        f"**signal_type**: {signal_type}  \n"
        f"**error_signature**: {error_sig or '_none detected_'}  \n"
        f"**stack_frames_count**: {len(frames)}\n",
    ]

    sections.append(_section("Source Locations", _fetch_source_locations(frames)))
    sections.append(_section("Related Tests", _fetch_related_tests(frames, test_path)))
    sections.append(_section("Recent Git Changes", _fetch_recent_git_changes(frames)))
    sections.append(_section("Launch Configurations", _fetch_launch_configs()))
    sections.append(_section("Debug Capabilities (Adapters)", _fetch_adapters()))

    return "\n".join(sections)


def create_build_debug_context_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=build_debug_context,
        name="build_debug_context",
        description=(
            "Build a structured debug context packet from a parsed signal. "
            "Fetches source code for stack frames (with confidence ratings), "
            "related tests, recent git changes around suspect files, available "
            "VS Code launch configs, and debug adapter capabilities. "
            "Pass stack_frames and error_signature from parse_debug_signal output."
        ),
        args_schema=BuildDebugContextInput,
    )
