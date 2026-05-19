"""Unit tests for run_validation tool (A6).

Tests cover all four status branches and auxiliary behaviours:
  1.  Success path: exit_code=0 → status=passed, evidence mentions wall time.
  2.  Failure path: exit_code=1, pytest-shaped output → status=failed, evidence
      carries "failed —" prefix and the first error-shaped line.
  3.  Failure path: exit_code=2, generic stderr "Error: something broke" →
      status=failed, evidence picks up that line.
  4.  Failure path, no error-shaped line → falls back to last non-empty line.
  5.  Timeout path: executor returns "timeout" error_type → status=timed_out,
      exit_code=None, evidence mentions the timeout limit.
  6.  Error path: no-tunnel error_type → status=error, evidence includes message.
  7.  output_excerpt is capped at ~800 chars.
  8.  evidence_summary is capped at 240 chars.
  9.  command field echoes the input.
  10. Tool registered in tool_service.py (AST source check).
  11. "run_validation" appears in DebugAgent's get_tools([...]) list (AST check).
"""

from __future__ import annotations

import os

# Set mandatory env vars before any app module is imported.
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from unittest.mock import patch, MagicMock

import pytest

from app.modules.intelligence.tools.run_validation_tool import (
    RunValidationInput,
    RunValidationResult,
    run_validation,
    run_validation_tool,
    _find_last_error_line,
    _last_nonempty_line,
    _build_evidence_summary,
    _MAX_EXCERPT_CHARS,
    _MAX_EVIDENCE_CHARS,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers — patch target for route_terminal_command
# ---------------------------------------------------------------------------

_PATCH_TARGET = (
    "app.modules.intelligence.tools.run_validation_tool.route_terminal_command"
)
_CONTEXT_PATCH = (
    "app.modules.intelligence.tools.run_validation_tool.get_context_vars"
)


def _make_result(
    *,
    success: bool = True,
    exit_code: int = 0,
    output: str = "",
    error: str = "",
    duration_ms: int = 4300,
) -> dict:
    return {
        "success": success,
        "exit_code": exit_code,
        "output": output,
        "error": error,
        "duration_ms": duration_ms,
        "command": "pytest",
        "truncated": False,
        "warnings": [],
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# 1. Success path: exit_code=0 → status=passed
# ---------------------------------------------------------------------------


def test_success_path_status_passed():
    result_dict = _make_result(success=True, exit_code=0, output="1 passed in 4.3s", duration_ms=4300)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/test_foo.py")

    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert "passed" in result["evidence_summary"].lower()
    assert result["command"] == "pytest tests/test_foo.py"


def test_success_path_evidence_mentions_wall_time():
    result_dict = _make_result(success=True, exit_code=0, duration_ms=4300)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/test_foo.py")

    # duration_ms=4300 → 4.3s in evidence
    assert "4.3s" in result["evidence_summary"]


def test_success_path_exit_code_populated():
    result_dict = _make_result(success=True, exit_code=0)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="npm test")

    assert result["exit_code"] == 0


# ---------------------------------------------------------------------------
# 2. Failure path: exit_code=1, pytest-shaped output
# ---------------------------------------------------------------------------

_PYTEST_OUTPUT = """\
FAILED tests/payments/test_payment_service.py::test_charge_returns_402_on_timeout - AssertionError: assert 500 == 402
  where 500 = <Response [500]>.status_code

tests/payments/test_payment_service.py:34: AssertionError
1 failed in 2.1s
"""


def test_failure_path_exit1_pytest_status_failed():
    result_dict = _make_result(success=False, exit_code=1, output=_PYTEST_OUTPUT)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/payments/test_payment_service.py")

    assert result["status"] == "failed"
    assert result["exit_code"] == 1


def test_failure_path_exit1_evidence_has_failed_prefix():
    result_dict = _make_result(success=False, exit_code=1, output=_PYTEST_OUTPUT)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/payments/test_payment_service.py")

    assert result["evidence_summary"].startswith("failed —")


def test_failure_path_exit1_evidence_contains_error_line():
    result_dict = _make_result(success=False, exit_code=1, output=_PYTEST_OUTPUT)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/payments/test_payment_service.py")

    # Should surface something from the FAILED line or AssertionError
    summary = result["evidence_summary"]
    assert "AssertionError" in summary or "FAILED" in summary or "assert" in summary.lower()


# ---------------------------------------------------------------------------
# 3. Failure path: exit_code=2, generic stderr "Error: something broke"
# ---------------------------------------------------------------------------

_GENERIC_STDERR = "Error: something broke\nsome other info"


def test_failure_path_exit2_generic_stderr_status_failed():
    result_dict = _make_result(success=False, exit_code=2, error=_GENERIC_STDERR)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="npm test")

    assert result["status"] == "failed"
    assert result["exit_code"] == 2


def test_failure_path_exit2_evidence_picks_error_line():
    result_dict = _make_result(success=False, exit_code=2, error=_GENERIC_STDERR)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="npm test")

    # "Error:" is in the pattern — should appear in evidence
    assert "Error" in result["evidence_summary"] or "broke" in result["evidence_summary"]


# ---------------------------------------------------------------------------
# 4. Failure path with no error-shaped line → last non-empty line fallback
# ---------------------------------------------------------------------------

_PLAIN_FAIL_OUTPUT = "Starting tests...\nAll tests completed.\nSome tests did not pass."


def test_failure_no_error_line_falls_back_to_last_nonempty():
    result_dict = _make_result(success=False, exit_code=1, output=_PLAIN_FAIL_OUTPUT)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="./run_tests.sh")

    # No pattern match → falls back to "Some tests did not pass."
    assert result["status"] == "failed"
    assert result["evidence_summary"].startswith("failed —")
    assert "Some tests did not pass" in result["evidence_summary"]


# ---------------------------------------------------------------------------
# 5. Timeout path
# ---------------------------------------------------------------------------


def test_timeout_status_is_timed_out():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest", timeout_seconds=30)

    assert result["status"] == "timed_out"


def test_timeout_exit_code_is_none():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest", timeout_seconds=30)

    assert result["exit_code"] is None


def test_timeout_evidence_mentions_limit():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest", timeout_seconds=120)

    assert "120" in result["evidence_summary"]
    assert "limit" in result["evidence_summary"].lower() or "timed out" in result["evidence_summary"].lower()


# ---------------------------------------------------------------------------
# 6. Error path: no-tunnel → status=error
# ---------------------------------------------------------------------------


def test_error_path_no_tunnel_status_error():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest")

    assert result["status"] == "error"
    assert result["exit_code"] is None


def test_error_path_evidence_includes_message():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest")

    assert "no_tunnel" in result["evidence_summary"] or "Could not run" in result["evidence_summary"]


def test_error_path_command_not_found():
    """Command-not-found scenario: executor raises exception."""
    with patch(_PATCH_TARGET, side_effect=Exception("command not found: pytest")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest nonexistent_module")

    assert result["status"] == "error"
    assert result["exit_code"] is None
    assert "command not found" in result["evidence_summary"]


def test_blocked_command_treated_as_error():
    result_dict = _make_result(success=False, exit_code=-1, error="command blocked by security policy")
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("u1", "c1")):
        result = run_validation(command="rm -rf /")
    assert result["status"] == "error"
    assert result["exit_code"] is None
    assert "blocked" in result["evidence_summary"].lower() or "could not run" in result["evidence_summary"].lower()


# ---------------------------------------------------------------------------
# 7. output_excerpt is capped at ~800 chars
# ---------------------------------------------------------------------------


def test_output_excerpt_capped():
    long_output = "x" * 2000
    result_dict = _make_result(success=True, exit_code=0, output=long_output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest")

    assert len(result["output_excerpt"]) <= _MAX_EXCERPT_CHARS


def test_output_excerpt_short_output_not_truncated():
    short_output = "1 passed in 0.1s"
    result_dict = _make_result(success=True, exit_code=0, output=short_output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest")

    assert result["output_excerpt"] == short_output


# ---------------------------------------------------------------------------
# 8. evidence_summary is capped at 240 chars
# ---------------------------------------------------------------------------


def test_evidence_summary_capped_at_240_chars():
    # A very long error line that would overflow 240 chars
    long_error_line = "AssertionError: " + "a" * 500
    long_output = long_error_line + "\n"
    result_dict = _make_result(success=False, exit_code=1, output=long_output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest")

    assert len(result["evidence_summary"]) <= _MAX_EVIDENCE_CHARS


def test_evidence_summary_max_constant_is_240():
    assert _MAX_EVIDENCE_CHARS == 240


# ---------------------------------------------------------------------------
# 9. command field echoes the input
# ---------------------------------------------------------------------------


def test_command_field_echoes_input():
    cmd = "npm test -- src/checkout/createOrder.test.ts"
    result_dict = _make_result(success=True, exit_code=0)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command=cmd)

    assert result["command"] == cmd


# ---------------------------------------------------------------------------
# Unit tests for internal helpers (fast, no mocking needed)
# ---------------------------------------------------------------------------


def test_find_last_error_line_finds_assertion_error():
    output = "running tests\nAssertionError: expected 402, got 500\nsome trailing text"
    line = _find_last_error_line(output)
    assert line is not None
    assert "AssertionError" in line


def test_find_last_error_line_finds_error_colon():
    output = "starting\nError: connection refused\nend"
    line = _find_last_error_line(output)
    assert line is not None
    assert "Error" in line


def test_find_last_error_line_finds_failed_line():
    output = "FAILED tests/foo.py::test_bar - AssertionError: x != y\n1 failed"
    line = _find_last_error_line(output)
    assert line is not None
    # Either the FAILED line or the assertion line
    assert "FAILED" in line or "AssertionError" in line


def test_find_last_error_line_returns_none_when_no_match():
    output = "all tests passed\n1 passed in 0.5s"
    line = _find_last_error_line(output)
    assert line is None


def test_last_nonempty_line_returns_last_non_blank():
    output = "line one\nline two\n\n  \nline three\n\n"
    result = _last_nonempty_line(output)
    assert result == "line three"


def test_last_nonempty_line_empty_string():
    result = _last_nonempty_line("")
    assert result == ""


def test_build_evidence_summary_passed():
    summary = _build_evidence_summary("passed", 4.3, "", 120)
    assert "passed" in summary.lower()
    assert "4.3s" in summary


def test_build_evidence_summary_timed_out():
    summary = _build_evidence_summary("timed_out", 120.0, "", 120)
    assert "120" in summary


def test_build_evidence_summary_error():
    summary = _build_evidence_summary("error", 0.1, "", 120, "command not found")
    assert "command not found" in summary


def test_build_evidence_summary_cap():
    long_line = "AssertionError: " + "x" * 400
    summary = _build_evidence_summary("failed", 1.0, long_line + "\n", 120)
    assert len(summary) <= _MAX_EVIDENCE_CHARS


# ---------------------------------------------------------------------------
# 10. Tool registration in tool_service.py (AST source check)
# ---------------------------------------------------------------------------


def test_run_validation_registered_in_tool_service():
    """run_validation_tool must be imported and registered in tool_service.py."""
    import pathlib

    tools_dir = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "tools"
    )
    source = (tools_dir / "tool_service.py").read_text(encoding="utf-8")

    assert "run_validation_tool" in source, (
        "tool_service.py must import run_validation_tool from run_validation_tool module"
    )
    assert '"run_validation"' in source or "'run_validation'" in source, (
        "tool_service.py must register the tool under the key 'run_validation'"
    )


# ---------------------------------------------------------------------------
# 11. DebugAgent tool list (AST inspection)
# ---------------------------------------------------------------------------


def test_run_validation_in_debug_agent_tool_list():
    """'run_validation' must appear in DebugAgent's get_tools([...]) call."""
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "agents"
        / "chat_agents"
        / "system_agents"
        / "debug_agent.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    tool_list: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_tools"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            for elt in node.args[0].elts:
                if isinstance(elt, ast.Constant):
                    tool_list.append(elt.value)

    assert tool_list, "Could not find get_tools([...]) call in debug_agent.py"
    assert "run_validation" in tool_list, (
        f"'run_validation' not found in DebugAgent's get_tools([...]) call. "
        f"Found: {tool_list}"
    )


# ---------------------------------------------------------------------------
# 12. Regression tests for bare FAILED/FAIL line anchor bug
# ---------------------------------------------------------------------------


def test_failed_alone_on_line_returns_failed_as_evidence():
    """FAILED followed by a newline must anchor to the FAILED line, not the next."""
    output = "some prelude\nFAILED\nnext line text\n"
    result_dict = _make_result(success=False, exit_code=1, output=output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="jest")

    summary = result["evidence_summary"]
    assert "FAILED" in summary
    assert "next line text" not in summary


def test_fail_alone_on_line_returns_fail_as_evidence():
    """FAIL followed by a newline must anchor to the FAIL line, not the next."""
    output = "running tests...\nFAIL\nbottom"
    result_dict = _make_result(success=False, exit_code=1, output=output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="jest")

    summary = result["evidence_summary"]
    assert "FAIL" in summary
    assert "bottom" not in summary


def test_failed_followed_by_space_still_works():
    """FAILED followed by a space (common pytest case) must still work correctly."""
    output = "FAILED tests/foo.py::test_bar - AssertionError: expected 1 got 2"
    result_dict = _make_result(success=False, exit_code=1, output=output)
    with patch(_PATCH_TARGET, return_value=(result_dict, None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = run_validation(command="pytest tests/foo.py")

    summary = result["evidence_summary"]
    # The line contains AssertionError — it should appear in evidence
    assert "AssertionError" in summary or "FAILED" in summary


# ---------------------------------------------------------------------------
# 13. RunValidationResult Pydantic round-trip
# ---------------------------------------------------------------------------


def test_result_pydantic_round_trip():
    r = RunValidationResult(
        status="passed",
        exit_code=0,
        evidence_summary="Command passed in 1.2s",
        output_excerpt="1 passed in 1.2s",
        wall_time_seconds=1.2,
        command="pytest",
    )
    d = r.model_dump()
    r2 = RunValidationResult.model_validate(d)
    assert r2.status == "passed"
    assert r2.exit_code == 0
    assert r2.command == "pytest"
