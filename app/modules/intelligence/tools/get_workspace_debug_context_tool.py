"""get_workspace_debug_context — workspace debug capabilities packet for the debug agent.

Returns the "debug capabilities" packet described in hypo.md §3.3:
  - launch configs from .vscode/launch.json
  - available debug adapter IDs in this VS Code session
  - recent git changes scoped to a file/directory or repo-wide
  - related test files for a given focus path
  - inferred fallback run/test commands from package.json / pyproject.toml

The backend dispatches an RPC named ``"workspace.debug_context"`` over the
existing tunnel/socket.io infrastructure (same pattern as ``route_terminal_command``
in tunnel_utils.py).  The extension-side handler is task E4 — until that lands
the tool returns ``available=False`` with a clear diagnostic message.

RPC route name: ``workspace.debug_context``
  - Chosen over alternatives (``debug.workspace_context``, ``workspace_debug_context``)
    because it clearly namespaces under the ``workspace`` domain and uses a
    period-separated sub-route style that mirrors Socket.IO event naming conventions.
  - The E4 TypeScript handler must register the same route string.

Dispatch approach: extended tunnel_utils.py with a focused
``route_workspace_debug_context`` function alongside ``route_terminal_command``.
This matches the "focused functions are easier to reason about" guidance and
avoids introducing a generic RPC helper that would require broader review.
"""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger
from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    route_workspace_debug_context,
    get_context_vars,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# A4.1 — Result schema (Pydantic)
# ---------------------------------------------------------------------------


class LaunchConfig(BaseModel):
    name: str = Field(..., description="Launch configuration name from .vscode/launch.json")
    type: str = Field(..., description="Debug adapter type (e.g. 'python', 'node', 'go').")
    request: str = Field(..., description="'launch' or 'attach'.")
    program: Optional[str] = Field(None, description="Entry point path if specified.")


class InferredCommand(BaseModel):
    label: str = Field(..., description="Short label, e.g. 'pytest', 'npm test', 'go test ./...'.")
    command: str = Field(..., description="The actual command string to run.")
    source: str = Field(..., description="Where it was inferred from, e.g. 'pyproject.toml', 'package.json scripts.test'.")


class RecentChange(BaseModel):
    file: str = Field(..., description="Workspace-relative path of the changed file.")
    commit_sha: str = Field(..., description="Short commit SHA that last touched this file.")
    commit_message: str = Field(..., description="First line of the commit message.")
    relative_time: str = Field(..., description="Humanized 'N hours ago'.")


class WorkspaceDebugContext(BaseModel):
    launch_configs: List[LaunchConfig] = Field(
        default_factory=list,
        description="Launch configurations parsed from .vscode/launch.json.",
    )
    debug_adapters: List[str] = Field(
        default_factory=list,
        description="Adapter ids available in this VS Code session (e.g. ['python', 'node']).",
    )
    recent_changes: List[RecentChange] = Field(
        default_factory=list,
        description="Recent git changes scoped to a file or to the whole repo.",
    )
    related_tests: List[str] = Field(
        default_factory=list,
        description="Test files that look related to the focus path.",
    )
    inferred_commands: List[InferredCommand] = Field(
        default_factory=list,
        description="Fallback run/test commands when .vscode/launch.json is sparse.",
    )
    available: bool = Field(
        ...,
        description="False when the extension is not attached or the tunnel could not deliver.",
    )
    message: Optional[str] = Field(
        None,
        description="Diagnostic message when available=False.",
    )


# ---------------------------------------------------------------------------
# A4.2 — Tool input schema
# ---------------------------------------------------------------------------


class GetWorkspaceDebugContextInput(BaseModel):
    focus_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional file or directory path within the workspace. "
            "When present, scopes related_tests and recent_changes to this path. "
            "When None (default), returns a repo-wide summary."
        ),
    )


# ---------------------------------------------------------------------------
# A4.2 — Core implementation
# ---------------------------------------------------------------------------

# Human-readable labels for known error_type values returned by
# route_workspace_debug_context.  Kept as module-level constants so tests can
# assert against them without duplicating strings.
_MSG_NO_TUNNEL = (
    "VS Code extension is not connected (no tunnel registered). "
    "Ensure the Potpie extension is running and shows 'Connected' status."
)
_MSG_TIMEOUT = (
    "workspace.debug_context RPC timed out — the VS Code extension may be busy "
    "or the tunnel is degraded. Retry when the extension is responsive."
)
_MSG_UNKNOWN_ROUTE = (
    "workspace.debug_context handler not yet implemented in the VS Code extension "
    "(E4 is pending). The backend can dispatch but the extension has not registered "
    "this route — returns available=False until E4 lands."
)
_MSG_TUNNEL_UNREACHABLE = (
    "Tunnel is registered but unreachable — the VS Code extension may have "
    "disconnected. Reconnect and retry."
)


def get_workspace_debug_context(
    focus_path: Optional[str] = None,
) -> dict:
    """Fetch workspace debug capabilities via tunnel RPC.

    Dispatches ``workspace.debug_context`` to the VS Code extension over the
    existing socket.io tunnel.  Returns a ``WorkspaceDebugContext`` dict.
    On any failure, returns ``available=False`` with a diagnostic message and
    empty lists — never raises.
    """
    user_id, conversation_id = get_context_vars()

    try:
        result, error_type = route_workspace_debug_context(
            focus_path=focus_path,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning(
            "[get_workspace_debug_context] Unexpected exception during dispatch: %s",
            exc,
        )
        return WorkspaceDebugContext(
            available=False,
            message=f"Unexpected error fetching workspace debug context: {exc}",
        ).model_dump()

    # --- Success: extension returned data ---
    if result is not None:
        try:
            # Deserialise each sub-field defensively; extension may return partial data.
            launch_configs = [
                LaunchConfig(**lc) for lc in (result.get("launch_configs") or [])
            ]
            debug_adapters: List[str] = result.get("debug_adapters") or []
            recent_changes = [
                RecentChange(**rc) for rc in (result.get("recent_changes") or [])
            ]
            related_tests: List[str] = result.get("related_tests") or []
            inferred_commands = [
                InferredCommand(**ic) for ic in (result.get("inferred_commands") or [])
            ]
            return WorkspaceDebugContext(
                launch_configs=launch_configs,
                debug_adapters=debug_adapters,
                recent_changes=recent_changes,
                related_tests=related_tests,
                inferred_commands=inferred_commands,
                available=True,
                message=None,
            ).model_dump()
        except Exception as parse_exc:
            logger.warning(
                "[get_workspace_debug_context] Failed to parse extension response: %s",
                parse_exc,
            )
            return WorkspaceDebugContext(
                available=False,
                message=f"Received a response from the extension but could not parse it: {parse_exc}",
            ).model_dump()

    # --- Failure: map error_type → human-readable message ---
    if error_type == "no_tunnel" or error_type == "no_user_id":
        message = _MSG_NO_TUNNEL
    elif error_type == "timeout":
        message = _MSG_TIMEOUT
    elif error_type == "unknown_route":
        message = _MSG_UNKNOWN_ROUTE
    elif error_type == "tunnel_unreachable":
        message = _MSG_TUNNEL_UNREACHABLE
    else:
        # Covers: "connection_error", "unknown_error", None, and any future codes.
        message = (
            f"workspace.debug_context RPC failed (error_type={error_type!r}). "
            "The VS Code extension tunnel may be unavailable or misconfigured."
        )

    logger.info(
        "[get_workspace_debug_context] returning available=False (error_type=%r)",
        error_type,
    )
    return WorkspaceDebugContext(
        available=False,
        message=message,
    ).model_dump()


# ---------------------------------------------------------------------------
# A4.3 — LangChain StructuredTool factory
# ---------------------------------------------------------------------------


def get_workspace_debug_context_tool() -> StructuredTool:
    """Create the get_workspace_debug_context StructuredTool.

    No db/user_id required — context vars are read inside the implementation.
    Factory shape matches run_validation_tool() (no-arg factory, context vars
    read internally).
    """
    return StructuredTool.from_function(
        func=get_workspace_debug_context,
        name="get_workspace_debug_context",
        description=(
            "Fetch the workspace debug capabilities packet from the connected VS Code "
            "extension via the tunnel RPC ('workspace.debug_context'). "
            "Returns launch configurations from .vscode/launch.json, available debug "
            "adapter IDs, recent git changes, test files related to a focus path, and "
            "inferred fallback run/test commands from package.json or pyproject.toml. "
            "Call this ONCE per debugging session to ground hypothesis generation in "
            "what is actually runnable in the workspace. "
            "When available=False the extension is not connected or the handler is not "
            "yet registered (E4 pending) — fall back to ask_knowledge_graph_queries, "
            "get_code_file_structure, or run_validation with inferred commands."
        ),
        args_schema=GetWorkspaceDebugContextInput,
    )
