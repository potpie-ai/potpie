"""DAP (Debug Adapter Protocol) Pydantic schemas for the A3 tool family.

Wire-format mapping from DebugSessionManager.ts (TypeScript) → Python field names:

TypeScript interface                  Python model          Notes
────────────────────────────────────────────────────────────────────────────────
DebugCallFrame.frame_id               frame_id              same (TS already snake_case)
DebugCallFrame.function               name                  TS uses 'function', Python uses 'name'
                                                            (kept for readability; wire key = 'name')
DebugCallFrame.file                   source_path           TS uses 'file', Python uses 'source_path'
DebugCallFrame.line                   line                  same

DebugSnapshot.paused_at              paused_at             TS: {file, line, function}; kept as-is
DebugSnapshot.call_stack             call_stack            same
DebugSnapshot.locals                 locals                same
DebugSnapshot.expression_results     expression_results    TS array of {expression,result?,error?}
DebugSnapshot.session_id             session_id            same
DebugSnapshot.status                 status                TS: 'paused'|'running'|'stopped' etc.

StartSessionResult.session_id        session_id            same
StartSessionResult.program           program               same
StartSessionResult.language          language              same
StartSessionResult.status            status                same

StopSessionResult.session_id         session_id            same
StopSessionResult.stopped            stopped (unused)      We remap to Literal["terminated","not_found"]

SetBreakpointsResult.file            file                  same
SetBreakpointsResult.breakpoints     breakpoints           array of {line,verified,actual_line?,message?}

ListSessionsResult.session_id        session_id            same
ListSessionsResult.program           program               same
ListSessionsResult.language          language              same
ListSessionsResult.status            status                same
ListSessionsResult.createdAt         created_at            camelCase→snake_case in Python

TrackedDebugSession (TS)             TrackedDebugSession   subset — only fields exposed to agent

The TS DebugSessionStatus enum is "initialized"|"running"|"paused"|"stopped".
We also accept "terminated" in the Python models for forward-compatibility.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# A3.1.1 — DebugCallFrame
# ---------------------------------------------------------------------------


class DebugCallFrame(BaseModel):
    """A single frame in a call stack.

    Mirrors TS DebugCallFrame: {frame_id, function, file, line}.
    Python renames 'function' → 'name' and 'file' → 'source_path' for clarity;
    the wire JSON from the extension uses 'function' and 'file' respectively.
    Pydantic aliases for 'function' and 'file' allow model_validate() to accept
    the raw TS dict directly.
    """

    model_config = ConfigDict(populate_by_name=True)

    frame_id: int = Field(..., description="Unique frame ID assigned by the debug adapter.")
    name: str = Field(..., alias="function", description="Function or method name at this frame.")
    source_path: Optional[str] = Field(None, alias="file", description="Absolute path to the source file, if available.")
    line: Optional[int] = Field(None, description="Source line number (1-based).")
    column: Optional[int] = Field(None, description="Source column number, if provided.")


# ---------------------------------------------------------------------------
# A3.1.2 — DebugSnapshot
# ---------------------------------------------------------------------------


class DebugSnapshot(BaseModel):
    """Full state snapshot captured at a DAP stopped event.

    Mirrors TS DebugSnapshot:
      paused_at: {file, line, function}
      call_stack: DebugCallFrame[]
      locals: Record<string, string>
      expression_results: Array<{expression, result?, error?}>
      session_id: string
      status: string

    The 'status' field reflects the TS DebugSessionStatus literals
    ("running" | "paused" | "initialized" | "stopped") plus "terminated"
    for forward-compatibility.
    """

    session_id: str = Field(..., description="Internal session UUID.")
    status: str = Field(
        ...,
        description="Session status at snapshot time: 'paused', 'running', 'stopped', 'initialized', or 'terminated'.",
    )
    paused_at: Optional[Dict[str, Any]] = Field(
        None,
        description="Location where execution is paused: {file, line, function}. Null when running.",
    )
    call_stack: List[DebugCallFrame] = Field(
        default_factory=list,
        description="Ordered list of stack frames (top frame first).",
    )
    locals: Dict[str, str] = Field(
        default_factory=dict,
        description="Local variable name → string representation at the top frame.",
    )
    expression_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Evaluated expression results: [{expression, result?, error?}]. "
            "Mirrors TS DebugSnapshot.expression_results exactly."
        ),
    )
    output: Optional[str] = Field(None, description="Any console/stdout output captured during the step.")


# ---------------------------------------------------------------------------
# A3.1.3 — StartSessionResult
# ---------------------------------------------------------------------------


class StartSessionResult(BaseModel):
    """Result of starting a new debug session.

    Mirrors TS StartSessionResult: {session_id, program, language, status}.
    """

    session_id: str = Field(..., description="UUID assigned to this debug session.")
    program: Optional[str] = Field(None, description="Absolute path to the program being debugged.")
    language: Optional[str] = Field(None, description="Language of the program, e.g. 'python'.")
    status: str = Field(
        ...,
        description="Session status after start: 'initialized', 'running', 'failed', etc.",
    )
    message: Optional[str] = Field(None, description="Optional diagnostic message.")


# ---------------------------------------------------------------------------
# A3.1.4 — SetBreakpointsResult
# ---------------------------------------------------------------------------


class SetBreakpointsResult(BaseModel):
    """Result of setting breakpoints in a file.

    Mirrors TS SetBreakpointsResult:
      {file, breakpoints: Array<{line, verified, actual_line?, message?}>}
    """

    session_id: Optional[str] = Field(None, description="Session the breakpoints were set on.")
    file: str = Field(..., description="Absolute path to the file where breakpoints were set.")
    breakpoints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "DAP breakpoint results: [{line, verified, actual_line?, message?}]. "
            "Shape matches the TS extension response exactly."
        ),
    )
    message: Optional[str] = Field(None, description="Optional diagnostic message.")


# ---------------------------------------------------------------------------
# A3.1.5 — TrackedDebugSession (agent-visible subset)
# ---------------------------------------------------------------------------


class TrackedDebugSession(BaseModel):
    """Lightweight session descriptor returned by list_debug_sessions.

    Mirrors TS ListSessionsResult: {session_id, program, language, status, createdAt}.
    'createdAt' is renamed to 'created_at' (snake_case); the alias allows
    model_validate() to accept the raw TS dict directly.
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., description="Session UUID.")
    program: Optional[str] = Field(None, description="Program being debugged.")
    language: Optional[str] = Field(None, description="Language, e.g. 'python'.")
    status: str = Field(
        ...,
        description="Current status: 'initialized', 'running', 'paused', 'stopped', or 'terminated'.",
    )
    created_at: Optional[str] = Field(None, alias="createdAt", description="ISO 8601 timestamp when the session was created.")


# ---------------------------------------------------------------------------
# A3.1.6 — ListSessionsResult
# ---------------------------------------------------------------------------


class ListSessionsResult(BaseModel):
    """Wrapper for listing all active debug sessions."""

    sessions: List[TrackedDebugSession] = Field(
        default_factory=list,
        description="All currently tracked debug sessions.",
    )


# ---------------------------------------------------------------------------
# A3.1.7 — EvaluateResult
# ---------------------------------------------------------------------------


class EvaluateResult(BaseModel):
    """Result of evaluating a single expression in a debug frame."""

    session_id: str = Field(..., description="Session the expression was evaluated in.")
    expression: str = Field(..., description="The expression that was evaluated.")
    value: str = Field(..., description="String representation of the evaluated value.")
    type: Optional[str] = Field(None, description="Type name returned by the debug adapter, if available.")


# ---------------------------------------------------------------------------
# A3.1.8 — StopSessionResult
# ---------------------------------------------------------------------------


class StopSessionResult(BaseModel):
    """Result of stopping a debug session.

    Mirrors TS StopSessionResult: {session_id, stopped: boolean}.
    We remap 'stopped: true' → status="terminated", not found → "not_found".
    """

    session_id: str = Field(..., description="UUID of the session that was stopped.")
    status: Literal["terminated", "not_found"] = Field(
        ...,
        description="'terminated' if the session was successfully stopped; 'not_found' if no matching session.",
    )
    message: Optional[str] = Field(None, description="Optional diagnostic message.")


# ---------------------------------------------------------------------------
# A3.1.9 — DapError (uniform error envelope)
# ---------------------------------------------------------------------------


class DapError(BaseModel):
    """Uniform error envelope returned by every DAP tool on failure.

    error_type values:
      no_tunnel         — VS Code extension is not connected
      unknown_route     — Extension does not yet handle this DAP route
      timeout           — RPC timed out
      tunnel_unreachable — Tunnel registered but socket/HTTP call failed
      backend_socket_error — Backend socket bridge failed before delivery
      extension_error   — Extension/debug adapter returned a DAP error
      no_user_id        — No user context available
      unknown_error     — Catch-all for unexpected exceptions
    """

    available: bool = Field(
        False,
        description="Always False for error envelopes.",
    )
    error: str = Field(
        ...,
        description="Short machine-readable error string.",
    )
    # error_type values: keep in sync with route_dap_command's returned error codes.
    # Add new codes here only when the dispatcher actually emits them.
    error_type: Literal[
        "no_tunnel",
        "unknown_route",
        "timeout",
        "tunnel_unreachable",
        "backend_socket_error",
        "extension_error",
        "no_user_id",
        "unknown_error",
    ] = Field(..., description="Categorised error type for programmatic handling.")
    message: str = Field(
        ...,
        description="Human-readable diagnostic message suitable for logging or agent display.",
    )
