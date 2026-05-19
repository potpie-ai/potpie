"""DAP tool family (A3) — ten tools that drive the VS Code debugger via the tunnel.

Each tool dispatches a RPC call to the VS Code extension's DebugSessionManager
using ``route_dap_command`` from ``tunnel_utils.py``.

RPC naming convention (for E4 reference):
  Registry key (Python)   Wire method (route_dap_command)  Socket event / HTTP endpoint
  ─────────────────────────────────────────────────────────────────────────────────────
  start_debug_session     start_session                    debug.start_session / /api/debug/start-session
  set_breakpoints         set_breakpoints                  debug.set_breakpoints / /api/debug/set-breakpoints
  take_debug_snapshot     snapshot                         debug.snapshot / /api/debug/snapshot
  step_over               step_over                        debug.step_over / /api/debug/step-over
  step_into               step_into                        debug.step_into / /api/debug/step-into
  step_out                step_out                         debug.step_out / /api/debug/step-out
  continue_execution      continue_execution               debug.continue_execution / /api/debug/continue-execution
  evaluate_expression     evaluate                         debug.evaluate / /api/debug/evaluate
  list_debug_sessions     list_sessions                    debug.list_sessions / /api/debug/list-sessions
  stop_debug_session      stop_session                     debug.stop_session / /api/debug/stop-session

All tools return .model_dump(mode="json") dicts. On any failure (including
missing tunnel) they return a DapError dict — they never raise.
"""

# E4 dependency: `evaluate` requires a new TS handler not present in current
# DebugSessionManager.ts. All other 9 methods map to existing DebugSessionManager
# methods.

from typing import List, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger
from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    route_dap_command,
    get_context_vars,
)
from app.modules.intelligence.tools.dap_schemas import (
    DebugCallFrame,
    DebugSnapshot,
    DapError,
    EvaluateResult,
    ListSessionsResult,
    SetBreakpointsResult,
    StartSessionResult,
    StopSessionResult,
    TrackedDebugSession,
)

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_TUNNEL_MSG = (
    "VS Code extension is not connected (no tunnel registered). "
    "Ensure the Potpie extension is running and shows 'Connected' status."
)
_UNKNOWN_ROUTE_MSG = (
    "The VS Code extension has not yet registered this DAP route. "
    "Extension-side handlers are part of E4 — returns available=False until E4 lands."
)
_TIMEOUT_MSG = (
    "DAP RPC timed out — the VS Code extension may be busy or the tunnel is degraded. "
    "Retry when the extension is responsive."
)
_TUNNEL_UNREACHABLE_MSG = (
    "Tunnel is registered but unreachable — the VS Code extension may have disconnected. "
    "Reconnect and retry."
)
_NO_USER_ID_MSG = "No authenticated user context — cannot route DAP command."


def _make_dap_error(error_type: Optional[str], context: str) -> DapError:
    """Map a route_dap_command error_type string to a DapError model."""
    if error_type == "no_tunnel":
        msg = _NO_TUNNEL_MSG
        etype: Literal[
            "no_tunnel", "unknown_route", "timeout",
            "tunnel_unreachable", "no_user_id", "unknown_error"
        ] = "no_tunnel"
    elif error_type == "no_user_id":
        msg = _NO_USER_ID_MSG
        etype = "no_user_id"
    elif error_type == "unknown_route":
        msg = _UNKNOWN_ROUTE_MSG
        etype = "unknown_route"
    elif error_type == "timeout":
        msg = _TIMEOUT_MSG
        etype = "timeout"
    elif error_type == "tunnel_unreachable":
        msg = _TUNNEL_UNREACHABLE_MSG
        etype = "tunnel_unreachable"
    else:
        msg = (
            f"DAP command '{context}' failed (error_type={error_type!r}). "
            "The VS Code extension tunnel may be unavailable or misconfigured."
        )
        etype = "unknown_error"
    return DapError(
        available=False,
        error=error_type or "unknown_error",
        error_type=etype,
        message=msg,
    )


def _parse_call_stack(raw: list) -> List[DebugCallFrame]:
    """Parse a raw list of call-frame dicts from the extension into DebugCallFrame models."""
    frames = []
    for f in raw:
        try:
            frames.append(
                DebugCallFrame(
                    frame_id=f.get("frame_id", 0),
                    name=f.get("function") or f.get("name") or "<unknown>",
                    source_path=f.get("file") or f.get("source_path"),
                    line=f.get("line"),
                    column=f.get("column"),
                )
            )
        except Exception as exc:
            logger.debug("[DAP] Could not parse call frame %r: %s", f, exc)
    return frames


def _build_snapshot(result: dict) -> DebugSnapshot:
    """Deserialise a raw extension response into a DebugSnapshot."""
    return DebugSnapshot(
        session_id=result.get("session_id", ""),
        status=result.get("status", "paused"),
        paused_at=result.get("paused_at"),
        call_stack=_parse_call_stack(result.get("call_stack") or []),
        locals=result.get("locals") or {},
        expression_results=result.get("expression_results") or [],
        output=result.get("output"),
    )


# ---------------------------------------------------------------------------
# 1. start_debug_session
# ---------------------------------------------------------------------------


class StartDebugSessionInput(BaseModel):
    program: str = Field(
        ...,
        description="Path to the program entry point (e.g. 'src/main.py' or '/abs/path/main.py').",
    )
    language: Literal["python", "node", "go", "unknown"] = Field(
        "python",
        description="Language of the program being debugged.",
    )
    mode: Literal["launch", "attach"] = Field(
        "launch",
        description="'launch' to start a new process; 'attach' to connect to a running process.",
    )
    port: Optional[int] = Field(
        None,
        description="Port to attach to (only relevant for attach mode).",
    )
    args: Optional[List[str]] = Field(
        None,
        description="Additional command-line arguments to pass to the program.",
    )


def start_debug_session(
    program: str,
    language: Literal["python", "node", "go", "unknown"] = "python",
    mode: Literal["launch", "attach"] = "launch",
    port: Optional[int] = None,
    args: Optional[List[str]] = None,
) -> dict:
    """Start a new VS Code debug session."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {
        "program": program,
        "language": language,
        "mode": mode,
    }
    if port is not None:
        payload["port"] = port
    if args is not None:
        payload["args"] = args

    try:
        result, error_type = route_dap_command(
            method="start_session",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[start_debug_session] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error starting debug session: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            return StartSessionResult(
                session_id=result.get("session_id", ""),
                program=result.get("program"),
                language=result.get("language"),
                status=result.get("status", "initialized"),
                message=result.get("message"),
            ).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[start_debug_session] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "start_session").model_dump(mode="json")


def start_debug_session_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=start_debug_session,
        name="start_debug_session",
        description=(
            "Start a new VS Code debug session for a program. "
            "Sends a 'start_session' RPC to the DebugSessionManager in the connected extension. "
            "Returns StartSessionResult with the session_id on success, or DapError when the "
            "extension is not connected. Call get_workspace_debug_context first to discover "
            "available launch configurations."
        ),
        args_schema=StartDebugSessionInput,
    )


# ---------------------------------------------------------------------------
# 2. set_breakpoints
# ---------------------------------------------------------------------------


class SetBreakpointsInput(BaseModel):
    file: str = Field(
        ...,
        description="Path to the source file (absolute or workspace-relative).",
    )
    lines: List[int] = Field(
        ...,
        description="List of 1-based line numbers where breakpoints should be set.",
    )
    condition: Optional[str] = Field(
        None,
        description="Optional conditional expression; the debugger only pauses when this evaluates to true.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Target session UUID. When omitted the most recently created session is used.",
    )


def set_breakpoints(
    file: str,
    lines: List[int],
    condition: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Set breakpoints in a source file for a debug session."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {"file": file, "lines": lines}
    if condition is not None:
        payload["condition"] = condition
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        result, error_type = route_dap_command(
            method="set_breakpoints",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[set_breakpoints] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error setting breakpoints: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            return SetBreakpointsResult(
                session_id=result.get("session_id") or session_id,
                file=result.get("file", file),
                breakpoints=result.get("breakpoints") or [],
                message=result.get("message"),
            ).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[set_breakpoints] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "set_breakpoints").model_dump(mode="json")


def set_breakpoints_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=set_breakpoints,
        name="set_breakpoints",
        description=(
            "Set breakpoints in a source file for an active debug session. "
            "Replaces all breakpoints in the file (DAP semantics). "
            "Returns SetBreakpointsResult showing which lines were verified by the adapter."
        ),
        args_schema=SetBreakpointsInput,
    )


# ---------------------------------------------------------------------------
# 3. take_debug_snapshot
# ---------------------------------------------------------------------------


class TakeDebugSnapshotInput(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Target session UUID. When omitted the most recently created session is used.",
    )
    expressions: Optional[List[str]] = Field(
        None,
        description="Optional list of expressions to evaluate and include in the snapshot.",
    )
    wait_for_stop: bool = Field(
        False,
        description=(
            "If True and the session is paused, first send 'continue' then wait for the next "
            "stopped event before capturing. If the session is already running, waits for the "
            "next stop."
        ),
    )
    timeout_seconds: int = Field(
        30,
        description="Maximum seconds to wait for a stopped event when wait_for_stop=True.",
    )


def take_debug_snapshot(
    session_id: Optional[str] = None,
    expressions: Optional[List[str]] = None,
    wait_for_stop: bool = False,
    timeout_seconds: int = 30,
) -> dict:
    """Capture a full state snapshot of the current debug position."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {
        "wait": wait_for_stop,
        "timeout": timeout_seconds * 1000,  # extension expects milliseconds
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if expressions is not None:
        payload["expressions"] = expressions

    try:
        result, error_type = route_dap_command(
            method="snapshot",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
            timeout=float(timeout_seconds) + 5,
        )
    except Exception as exc:
        logger.warning("[take_debug_snapshot] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error taking snapshot: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            return _build_snapshot(result).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[take_debug_snapshot] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "snapshot").model_dump(mode="json")


def take_debug_snapshot_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=take_debug_snapshot,
        name="take_debug_snapshot",
        description=(
            "Capture a full snapshot of the current debugger state: call stack, local variables, "
            "and optionally evaluated expressions. The session must be paused unless wait_for_stop=True. "
            "Returns a DebugSnapshot or DapError."
        ),
        args_schema=TakeDebugSnapshotInput,
    )


# ---------------------------------------------------------------------------
# 4–6. Stepping tools (step_over / step_into / step_out)
# ---------------------------------------------------------------------------


class StepInput(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Target session UUID. When omitted the most recently created session is used.",
    )
    expressions: Optional[List[str]] = Field(
        None,
        description="Optional list of expressions to evaluate and include in the post-step snapshot.",
    )


def _step_tool_impl(method: str, session_id: Optional[str], expressions: Optional[List[str]]) -> dict:
    """Shared implementation for step_over / step_into / step_out."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {}
    if session_id is not None:
        payload["session_id"] = session_id
    if expressions is not None:
        payload["expressions"] = expressions

    try:
        result, error_type = route_dap_command(
            method=method,
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[%s] Unexpected exception: %s", method, exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error during {method}: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            return _build_snapshot(result).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[%s] Parse error: %s", method, parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, method).model_dump(mode="json")


def step_over(
    session_id: Optional[str] = None,
    expressions: Optional[List[str]] = None,
) -> dict:
    """Step over the current source line and return a snapshot of the new position."""
    return _step_tool_impl("step_over", session_id, expressions)


def step_into(
    session_id: Optional[str] = None,
    expressions: Optional[List[str]] = None,
) -> dict:
    """Step into the function call on the current line and return a snapshot."""
    return _step_tool_impl("step_into", session_id, expressions)


def step_out(
    session_id: Optional[str] = None,
    expressions: Optional[List[str]] = None,
) -> dict:
    """Step out of the current function and return a snapshot of the caller."""
    return _step_tool_impl("step_out", session_id, expressions)


def step_over_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=step_over,
        name="step_over",
        description=(
            "Step over (next line) in the active debug session. Session must be paused. "
            "Returns a DebugSnapshot after the step, or DapError on failure."
        ),
        args_schema=StepInput,
    )


def step_into_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=step_into,
        name="step_into",
        description=(
            "Step into the function call on the current line. Session must be paused. "
            "Returns a DebugSnapshot after stepping into the call, or DapError on failure."
        ),
        args_schema=StepInput,
    )


def step_out_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=step_out,
        name="step_out",
        description=(
            "Step out of the current function back to the caller. Session must be paused. "
            "Returns a DebugSnapshot at the return site, or DapError on failure."
        ),
        args_schema=StepInput,
    )


# ---------------------------------------------------------------------------
# 7. continue_execution
# ---------------------------------------------------------------------------


class ContinueExecutionInput(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Target session UUID. When omitted the most recently created session is used.",
    )


def continue_execution(session_id: Optional[str] = None) -> dict:
    """Resume execution of the paused debug session."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {}
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        result, error_type = route_dap_command(
            method="continue_execution",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[continue_execution] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error during continue_execution: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            # continueExecution returns {session_id, status} not a full snapshot
            return DebugSnapshot(
                session_id=result.get("session_id", session_id or ""),
                status=result.get("status", "running"),
                paused_at=None,
            ).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[continue_execution] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "continue_execution").model_dump(mode="json")


def continue_execution_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=continue_execution,
        name="continue_execution",
        description=(
            "Resume execution from the current breakpoint. The session must be paused. "
            "Returns immediately with status='running' (use take_debug_snapshot with "
            "wait_for_stop=True to wait for the next breakpoint)."
        ),
        args_schema=ContinueExecutionInput,
    )


# ---------------------------------------------------------------------------
# 8. evaluate_expression
# ---------------------------------------------------------------------------


class EvaluateExpressionInput(BaseModel):
    expression: str = Field(
        ...,
        description="The expression to evaluate in the current debug frame (REPL context).",
    )
    session_id: Optional[str] = Field(
        None,
        description="Target session UUID. When omitted the most recently created session is used.",
    )
    frame_id: Optional[int] = Field(
        None,
        description="Stack frame ID to evaluate the expression in. Defaults to the top frame.",
    )


def evaluate_expression(
    expression: str,
    session_id: Optional[str] = None,
    frame_id: Optional[int] = None,
) -> dict:
    """Evaluate a single expression in the current debug frame.

    Note for E4 (extension-side implementation):
        DebugSessionManager.ts does not yet have an ``evaluate(...)`` public method.
        The extension-side handler for /api/debug/evaluate must be added
        specifically for this tool. Expected response shape:
            {session_id: string, expression: string, result: string, type?: string}
    """
    user_id, conversation_id = get_context_vars()
    payload: dict = {"expression": expression}
    if session_id is not None:
        payload["session_id"] = session_id
    if frame_id is not None:
        payload["frame_id"] = frame_id

    try:
        result, error_type = route_dap_command(
            method="evaluate",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[evaluate_expression] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error evaluating expression: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            return EvaluateResult(
                session_id=result.get("session_id", session_id or ""),
                expression=result.get("expression", expression),
                value=str(result.get("result") or result.get("value") or ""),
                type=result.get("type"),
            ).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[evaluate_expression] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "evaluate").model_dump(mode="json")


def evaluate_expression_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=evaluate_expression,
        name="evaluate_expression",
        description=(
            "Evaluate a Python/JS/Go expression in the context of the current debug frame (REPL). "
            "Session must be paused. Returns EvaluateResult with the value and optional type, "
            "or DapError on failure."
        ),
        args_schema=EvaluateExpressionInput,
    )


# ---------------------------------------------------------------------------
# 9. list_debug_sessions
# ---------------------------------------------------------------------------


class ListDebugSessionsInput(BaseModel):
    pass  # No inputs required


def list_debug_sessions() -> dict:
    """List all active debug sessions tracked by DebugSessionManager."""
    user_id, conversation_id = get_context_vars()

    try:
        result, error_type = route_dap_command(
            method="list_sessions",
            payload={},
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[list_debug_sessions] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error listing debug sessions: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            raw_sessions = result if isinstance(result, list) else result.get("sessions") or []
            sessions = [
                TrackedDebugSession.model_validate(s)
                for s in raw_sessions
            ]
            return ListSessionsResult(sessions=sessions).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[list_debug_sessions] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "list_sessions").model_dump(mode="json")


def list_debug_sessions_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=list_debug_sessions,
        name="list_debug_sessions",
        description=(
            "List all currently active debug sessions tracked by the VS Code extension. "
            "Returns ListSessionsResult with session UUIDs, programs, languages, and statuses. "
            "Use this to discover session IDs before calling step/snapshot/evaluate tools."
        ),
        args_schema=ListDebugSessionsInput,
    )


# ---------------------------------------------------------------------------
# 10. stop_debug_session
# ---------------------------------------------------------------------------


class StopDebugSessionInput(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="UUID of the session to stop. When omitted the most recently created session is stopped.",
    )


def stop_debug_session(session_id: Optional[str] = None) -> dict:
    """Stop (terminate) an active debug session."""
    user_id, conversation_id = get_context_vars()
    payload: dict = {}
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        result, error_type = route_dap_command(
            method="stop_session",
            payload=payload,
            user_id=user_id or "",
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning("[stop_debug_session] Unexpected exception: %s", exc)
        return DapError(
            available=False,
            error="unknown_error",
            error_type="unknown_error",
            message=f"Unexpected error stopping debug session: {exc}",
        ).model_dump(mode="json")

    if result is not None:
        try:
            # TS StopSessionResult: {session_id, stopped: boolean}
            stopped = result.get("stopped", True)
            return StopSessionResult(
                session_id=result.get("session_id", session_id or ""),
                status="terminated" if stopped else "not_found",
                message=result.get("message"),
            ).model_dump(mode="json")
        except Exception as parse_exc:
            logger.warning("[stop_debug_session] Parse error: %s", parse_exc)
            return DapError(
                available=False,
                error="unknown_error",
                error_type="unknown_error",
                message=f"Received a response but could not parse it: {parse_exc}",
            ).model_dump(mode="json")

    return _make_dap_error(error_type, "stop_session").model_dump(mode="json")


def stop_debug_session_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=stop_debug_session,
        name="stop_debug_session",
        description=(
            "Stop and terminate an active debug session, disconnecting from the debuggee. "
            "Returns StopSessionResult with status='terminated' on success, or DapError on failure."
        ),
        args_schema=StopDebugSessionInput,
    )
