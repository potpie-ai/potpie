"""run_validation — structured pass/fail validation tool for the debug agent.

Runs a user-supplied command (typically a test or debug re-run), captures the
result via the existing execute_terminal_command plumbing, and returns a
structured pass/fail summary with evidence the agent can use for "before fix /
after fix" validation (hypo.md §5 Step 9).

This is distinct from generic bash execution: it adds structured outcome
semantics (status, exit_code, evidence_summary, output_excerpt, wall_time).
No direct subprocess usage — all execution is delegated to route_terminal_command.
"""

import re
import time
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger
from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    route_terminal_command,
    get_context_vars,
)

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled failure-line patterns (module-level, compiled once at import time).
# Anchored / left-anchored patterns to avoid catastrophic backtracking.
# ---------------------------------------------------------------------------

# Matches lines that begin with (or contain after leading whitespace) a
# recognisable error/failure keyword.
# Groups of alternatives ordered from most specific to least specific.
_FAILURE_LINE_RE = re.compile(
    r"""
    (?:^|\n)            # start of string or start of new line
    [^\S\n]*            # optional horizontal whitespace (tabs, spaces)
    (?:
        AssertionError  |
        AssertError     |
        assert\s+       |   # "assert x == y" lines from pytest
        Exception       |
        Error:          |   # "Error: something" (JS/TS style)
        FAILED(?=[^\S\n]|\n|$)  |   # "FAILED tests/..." pytest; bare "FAILED" jest EOL
        FAIL(?=[^\S\n]|\n|$)    |   # "FAIL tests/..." jest-style
        FailError       |
        Traceback       |   # Python traceback header
        SyntaxError     |
        TypeError       |
        ValueError      |
        KeyError        |
        AttributeError  |
        ImportError     |
        RuntimeError    |
        NotImplementedError |
        PermissionError |
        TimeoutError    |
        ConnectionError |
        OSError         |
        E\s{5,}         |   # pytest inline error lines (E       AssertionError)
        panic:          |   # Go panic
        PANIC           |   # Go PANIC
        fatal\s+error   |   # Go / C fatal
        \[FAIL\]        |   # Go test output
        ● \s            |   # Jest/Mocha bullet marker
        ✕ \s            |   # Vitest cross marker
        ✗ \s                # alternative cross marker
    )
    """,
    re.VERBOSE | re.MULTILINE,
)

# Maximum characters for output_excerpt
_MAX_EXCERPT_CHARS: int = 800
# Maximum characters for evidence_summary
_MAX_EVIDENCE_CHARS: int = 240


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RunValidationInput(BaseModel):
    command: str = Field(
        description=(
            "The shell command to run for validation, e.g. "
            "'npm test -- src/checkout/createOrder.test.ts' or "
            "'pytest tests/test_checkout.py -k payment_timeout'"
        )
    )
    working_directory: Optional[str] = Field(
        default=None,
        description=(
            "Working directory relative to workspace root (optional). "
            "Defaults to the project root via the existing terminal-command convention."
        ),
    )
    timeout_seconds: int = Field(
        default=120,
        description="Wall-clock cap in seconds before the process is killed (default: 120).",
    )


class RunValidationResult(BaseModel):
    status: Literal["passed", "failed", "timed_out", "error"]
    exit_code: Optional[int]
    evidence_summary: str
    output_excerpt: str
    wall_time_seconds: float
    command: str


# ---------------------------------------------------------------------------
# Evidence-summary helpers
# ---------------------------------------------------------------------------


def _find_last_error_line(output: str) -> Optional[str]:
    """Scan *output* bottom-up for the last line matching a failure pattern.

    Returns the stripped matching line, or None if nothing matches.
    """
    if not output:
        return None

    # Collect all match positions
    matches = list(_FAILURE_LINE_RE.finditer(output))
    if not matches:
        return None

    # Take the LAST match (bottom-up scan gives us the most relevant line,
    # typically the final error before the summary)
    match = matches[-1]

    # The regex may include a leading `\n` (or `(?:^|\n)`) in the match.
    # Use match.end() as the anchor: it points past the keyword.  Then find
    # the full line by searching backwards for the preceding newline and
    # forwards for the next newline.
    anchor = match.end()
    # Find the start of the line that contains the keyword
    line_start = output.rfind("\n", 0, anchor) + 1  # +1 skips the \n itself
    line_end = output.find("\n", anchor)
    if line_end == -1:
        line_end = len(output)
    return output[line_start:line_end].strip()


def _last_nonempty_line(output: str) -> str:
    """Return the last non-empty stripped line, or empty string."""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _build_evidence_summary(
    status: str,
    wall_time: float,
    output: str,
    timeout_seconds: int,
    error_msg: str = "",
) -> str:
    """Construct a ≤240-char human-readable evidence summary."""
    if status == "passed":
        summary = f"Command passed in {wall_time:.1f}s"

    elif status == "failed":
        error_line = _find_last_error_line(output)
        if not error_line:
            error_line = _last_nonempty_line(output)
        if error_line:
            summary = f"failed — {error_line}"
        else:
            summary = f"failed — exit code non-zero in {wall_time:.1f}s"

    elif status == "timed_out":
        summary = f"Timed out after {timeout_seconds}s (limit reached)"

    else:  # error
        if error_msg:
            summary = f"Could not run: {error_msg}"
        else:
            summary = "Could not run: unknown execution error"

    return summary[:_MAX_EVIDENCE_CHARS]


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def run_validation(
    command: str,
    working_directory: Optional[str] = None,
    timeout_seconds: int = 120,
) -> dict:
    """Run *command* and return a structured pass/fail result.

    Delegates execution to route_terminal_command (the same underlying
    executor used by execute_terminal_command_tool) — no direct subprocess.
    """
    user_id, conversation_id = get_context_vars()

    # Convert timeout_seconds → milliseconds for route_terminal_command
    timeout_ms = timeout_seconds * 1000

    t_start = time.monotonic()

    try:
        result, error_type = route_terminal_command(
            command=command,
            working_directory=working_directory,
            timeout=timeout_ms,
            mode="sync",
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        wall_time = time.monotonic() - t_start
        err_msg = str(exc)
        logger.warning(f"[run_validation] Unexpected exception executing command: {exc}")
        summary = _build_evidence_summary("error", wall_time, "", timeout_seconds, err_msg)
        return RunValidationResult(
            status="error",
            exit_code=None,
            evidence_summary=summary,
            output_excerpt=err_msg[:_MAX_EXCERPT_CHARS],
            wall_time_seconds=round(wall_time, 3),
            command=command,
        ).model_dump()

    wall_time = time.monotonic() - t_start

    # If route_terminal_command used duration_ms from the remote executor, prefer it
    if result and result.get("duration_ms") is not None:
        wall_time = result["duration_ms"] / 1000.0

    # -----------------------------------------------------------------------
    # Determine status
    # -----------------------------------------------------------------------

    if error_type == "timeout" or (result and result.get("timed_out")):
        # Timed out
        status: Literal["passed", "failed", "timed_out", "error"] = "timed_out"
        exit_code: Optional[int] = None
        combined_output = ""
        if result:
            combined_output = (result.get("output") or "") + (result.get("error") or "")
        output_excerpt = combined_output[:_MAX_EXCERPT_CHARS]
        summary = _build_evidence_summary("timed_out", wall_time, "", timeout_seconds)

    # Invariant: route_terminal_command returns result=None whenever error_type is set,
    # so the fall-through "result is None" branch catches any future unrecognized error types.
    elif result is None or error_type in (
        "no_tunnel",
        "no_user_id",
        "tunnel_unreachable",
        "tunnel_expired",
        "connection_error",
        "unknown_error",
    ):
        # Could not run at all
        status = "error"
        exit_code = None
        err_msg = error_type or "no result returned"
        output_excerpt = err_msg[:_MAX_EXCERPT_CHARS]
        summary = _build_evidence_summary("error", wall_time, "", timeout_seconds, err_msg)

    else:
        # Command ran to completion (success or failure)
        raw_exit = result.get("exit_code")
        # The extension may report -1 for "blocked" commands
        if raw_exit == -1 and not result.get("success", True):
            # Blocked by security policy → treat as error
            status = "error"
            exit_code = None
            err_msg = result.get("error", "command blocked by security policy")
            combined_output = (result.get("output") or "") + (result.get("error") or "")
            output_excerpt = combined_output[:_MAX_EXCERPT_CHARS]
            summary = _build_evidence_summary("error", wall_time, combined_output, timeout_seconds, err_msg)
        elif raw_exit == 0:
            status = "passed"
            exit_code = 0
            combined_output = (result.get("output") or "") + (result.get("error") or "")
            output_excerpt = combined_output[:_MAX_EXCERPT_CHARS]
            summary = _build_evidence_summary("passed", wall_time, combined_output, timeout_seconds)
        else:
            status = "failed"
            exit_code = raw_exit if raw_exit is not None else 1
            combined_output = (result.get("output") or "") + (result.get("error") or "")
            output_excerpt = combined_output[:_MAX_EXCERPT_CHARS]
            summary = _build_evidence_summary("failed", wall_time, combined_output, timeout_seconds)

    return RunValidationResult(
        status=status,
        exit_code=exit_code,
        evidence_summary=summary,
        output_excerpt=output_excerpt,
        wall_time_seconds=round(wall_time, 3),
        command=command,
    ).model_dump()


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory
# ---------------------------------------------------------------------------


def run_validation_tool() -> StructuredTool:
    """Create the run_validation StructuredTool.

    No db/user_id required — context vars are read inside run_validation.
    Mirrors the parse_failure_signal_tool() factory signature.
    """
    return StructuredTool.from_function(
        func=run_validation,
        name="run_validation",
        description=(
            "Run a shell command (typically a test suite or a debug re-run) and "
            "return a structured pass/fail summary. "
            "Use this tool for the 'before fix / after fix' validation step to "
            "produce machine-readable evidence that a fix worked. "
            "Unlike generic bash execution, this tool returns a structured result "
            "with status ('passed' | 'failed' | 'timed_out' | 'error'), exit_code, "
            "a short evidence_summary (≤240 chars) suitable for inline display, "
            "an output_excerpt (≤800 chars) for deeper inspection, and wall_time_seconds. "
            "Examples: 'npm test -- src/checkout/createOrder.test.ts', "
            "'pytest tests/test_checkout.py -k payment_timeout', "
            "'go test ./pkg/payments/... -run TestChargeTimeout'."
        ),
        args_schema=RunValidationInput,
    )
