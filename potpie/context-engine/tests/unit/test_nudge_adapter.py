"""Step 12b: the model-free nudge hook adapter (`potpie_nudge.py`).

The adapter is a standalone, stdlib-only script shipped inside the Claude Code
plugin (and reused by Codex/Cursor). These tests load it directly from its template
path and exercise:

- mechanical event mapping (harness hint + payload → nudge event), including the
  ``PostToolUse(Bash)`` → ``test_failed`` / ``test_passed`` split that the harness
  taxonomy forces;
- command classification (deploy / test) and test-outcome inference;
- argv construction and harness-output rendering;
- end-to-end fail-safety: a real subprocess run with a fake ``potpie`` injects, and a
  missing binary / empty stdin / broken output never errors or emits noise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import adapters.inbound.cli as _clipkg

pytestmark = pytest.mark.unit

ADAPTER_PATH = (
    Path(_clipkg.__file__).resolve().parent
    / "templates"
    / "claude_plugin"
    / "hooks"
    / "potpie_nudge.py"
)


def _load_adapter():
    # Compile + exec in-memory so no ``__pycache__`` is written into the templates
    # tree (which the installer reads as text).
    src = ADAPTER_PATH.read_text(encoding="utf-8")
    mod = types.ModuleType("potpie_nudge_under_test")
    mod.__file__ = str(ADAPTER_PATH)
    exec(compile(src, str(ADAPTER_PATH), "exec"), mod.__dict__)
    return mod


adapter = _load_adapter()


# --- payload accessors ------------------------------------------------------


def test_session_id_prefers_payload_then_env(monkeypatch) -> None:
    assert adapter.session_id_of({"session_id": "abc"}) == "abc"
    assert adapter.session_id_of({"sessionId": "xyz"}) == "xyz"
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.delenv("POTPIE_SESSION_ID", raising=False)
    assert adapter.session_id_of({}) == "default"
    monkeypatch.setenv("CLAUDE_SESSION_ID", "env-sess")
    assert adapter.session_id_of({}) == "env-sess"


def test_file_path_and_command_accessors_tolerate_shapes() -> None:
    assert adapter.file_path_of({"tool_input": {"file_path": "a.py"}}) == "a.py"
    assert adapter.file_path_of({"toolInput": {"file_path": "b.py"}}) == "b.py"
    assert adapter.file_path_of({"path": "c.py"}) == "c.py"
    assert adapter.file_path_of({}) is None
    assert adapter.command_of({"tool_input": {"command": "ls"}}) == "ls"
    assert adapter.command_of({"command": "pwd"}) == "pwd"
    assert adapter.command_of({}) is None


# --- command classification -------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "kubectl apply -f deploy.yaml",
        "helm upgrade api ./chart",
        "terraform apply",
        "fly deploy",
    ],
)
def test_is_deploy_command_true(command: str) -> None:
    assert adapter.is_deploy_command(command) is True


@pytest.mark.parametrize("command", ["ls -la", "git status", "echo hi", None, ""])
def test_is_deploy_command_false(command) -> None:
    assert adapter.is_deploy_command(command) is False


@pytest.mark.parametrize(
    "command", ["pytest -q", "npm test", "go test ./...", "cargo test", "make check"]
)
def test_is_test_command_true(command: str) -> None:
    assert adapter.is_test_command(command) is True


@pytest.mark.parametrize("command", ["python app.py", "ruff check", None])
def test_is_test_command_false(command) -> None:
    assert adapter.is_test_command(command) is False


def test_test_outcome_prefers_exit_code() -> None:
    assert adapter.test_outcome({"exit_code": 0, "stderr": "AssertionError"}) == "pass"
    assert adapter.test_outcome({"exit_code": 1, "stdout": "all passed"}) == "fail"
    assert adapter.test_outcome({"returncode": 2}) == "fail"


def test_test_outcome_falls_back_to_signatures() -> None:
    assert adapter.test_outcome({"stdout": "5 passed in 1.2s"}) == "pass"
    assert adapter.test_outcome({"stderr": "1 failed, 2 passed"}) == "fail"
    assert adapter.test_outcome({"is_error": True}) == "fail"


def test_test_outcome_ambiguous_or_interrupted_is_none() -> None:
    assert adapter.test_outcome({"stdout": "running tests..."}) is None
    assert adapter.test_outcome({"interrupted": True, "exit_code": 1}) is None
    assert adapter.test_outcome({}) is None
    assert adapter.test_outcome("") is None


# Regression: the live Claude Code Bash path carries NO exit code, so classification
# falls to text. These are real green/red outputs that the naive substring approach
# misclassified ("0 failed" / "error" in names shadowing PASS).
@pytest.mark.parametrize(
    "text,expected",
    [
        # green runs that report an explicit zero failure count → pass (not fail)
        ("test result: ok. 12 passed; 0 failed; 0 ignored", "pass"),  # cargo
        ("Tests: 42 passed, 0 failed, 42 total", "pass"),  # jest
        ("ok\tgithub.com/acme/x\t0.012s", "pass"),  # go (tab-separated ok line)
        ("ok github.com/acme/errors 0.01s\nPASS", "pass"),  # 'errors' in pkg path
        (
            "test_error_handling PASSED\n1 passed in 0.1s",
            "pass",
        ),  # 'error' in test name
        ("42 passing (1s)", "pass"),  # mocha
        # red runs → fail
        ("Tests: 3 failed, 39 passed, 42 total", "fail"),  # jest
        ("test result: FAILED. 10 passed; 2 failed", "fail"),  # cargo
        ("1 failed, 2 passed in 0.5s", "fail"),  # pytest
        ("3 failing\n  1) settles deadlock", "fail"),  # mocha
        (
            "Traceback (most recent call last):\nAssertionError: boom",
            "fail",
        ),  # no count
        ("--- FAIL: TestSettle (0.00s)\nFAIL", "fail"),  # go -v
    ],
)
def test_test_outcome_real_world_outputs(text: str, expected: str) -> None:
    assert adapter.test_outcome({"stdout": text}) == expected


# --- event resolution -------------------------------------------------------


def test_resolve_session_start_and_stop() -> None:
    assert adapter.resolve_nudge_event("session_start", {}) == ("session_start", {})
    assert adapter.resolve_nudge_event("stop", {}) == ("stop", {})


def test_resolve_pre_edit_carries_path() -> None:
    ev, fields = adapter.resolve_nudge_event(
        "pre_edit",
        {"tool_name": "Edit", "tool_input": {"file_path": "src/payments/client.py"}},
    )
    assert ev == "pre_edit"
    assert fields["path"] == "src/payments/client.py"


def test_direct_dash_event_alias_is_canonicalized() -> None:
    ev, fields = adapter.resolve_nudge_event(
        "pre-edit",
        {"tool_name": "Edit", "tool_input": {"file_path": "src/payments/client.py"}},
    )
    assert ev == "pre_edit"
    assert fields == {"path": "src/payments/client.py", "query": None}


def test_resolve_bash_pre_deploy_vs_skip() -> None:
    ev, fields = adapter.resolve_nudge_event(
        "bash_pre", {"tool_input": {"command": "kubectl apply -f svc.yaml"}}
    )
    assert ev == "pre_deploy" and "kubectl apply" in fields["query"]
    # Non-deploy bash must NOT nudge (noise control).
    assert adapter.resolve_nudge_event(
        "bash_pre", {"tool_input": {"command": "ls"}}
    ) == (
        None,
        {},
    )


def test_resolve_bash_post_test_failed_extracts_symptom() -> None:
    ev, fields = adapter.resolve_nudge_event(
        "bash_post",
        {
            "tool_input": {"command": "pytest -q tests/test_settle.py"},
            "tool_response": {
                "exit_code": 1,
                "stderr": "E   AssertionError: deadlock on settle",
            },
        },
    )
    assert ev == "test_failed"
    assert "AssertionError" in fields["query"] or "settle" in fields["query"]


def test_resolve_bash_post_test_passed() -> None:
    ev, fields = adapter.resolve_nudge_event(
        "bash_post",
        {
            "tool_input": {"command": "pytest -q"},
            "tool_response": {"exit_code": 0, "stdout": "5 passed"},
        },
    )
    assert ev == "test_passed"
    assert fields["query"] == "pytest -q"


def test_resolve_bash_post_non_test_and_ambiguous_skip() -> None:
    assert adapter.resolve_nudge_event(
        "bash_post",
        {"tool_input": {"command": "ls"}, "tool_response": {"exit_code": 0}},
    ) == (None, {})
    # A test command with no determinable outcome stays silent rather than nudging.
    assert adapter.resolve_nudge_event(
        "bash_post",
        {
            "tool_input": {"command": "pytest"},
            "tool_response": {"stdout": "collecting"},
        },
    ) == (None, {})


def test_resolve_unknown_hint_is_silent() -> None:
    assert adapter.resolve_nudge_event("nonsense", {}) == (None, {})


# --- argv + output rendering ------------------------------------------------


def test_build_argv_shape() -> None:
    argv = adapter.build_argv(
        "potpie",
        "pre_edit",
        "sess-1",
        path="a.py",
        query="q",
        pot="local/default",
        limit=5,
    )
    assert argv[:5] == ["potpie", "--json", "graph", "nudge", "--event"]
    assert "pre_edit" in argv and "--session" in argv and "sess-1" in argv
    assert "--path" in argv and "a.py" in argv
    assert "--pot" in argv and "local/default" in argv
    assert "--limit" in argv and "5" in argv


def test_build_argv_omits_absent_optionals() -> None:
    argv = adapter.build_argv("potpie", "stop", "s")
    assert "--path" not in argv and "--query" not in argv and "--pot" not in argv


def test_render_output_silent_and_not_ok_emit_nothing() -> None:
    assert adapter.render_output("PreToolUse", {"ok": True, "silent": True}) == ("", 0)
    assert adapter.render_output(
        "PreToolUse", {"ok": False, "inject_context": "x"}
    ) == ("", 0)
    assert adapter.render_output("PreToolUse", {"ok": True, "silent": False}) == ("", 0)
    assert adapter.render_output("PreToolUse", "not-a-dict") == ("", 0)


def test_render_output_injects_context_envelope() -> None:
    out, code = adapter.render_output(
        "PreToolUse",
        {"ok": True, "silent": False, "inject_context": "PREF: use tenacity"},
    )
    assert code == 0
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "tenacity" in parsed["hookSpecificOutput"]["additionalContext"]


def test_render_output_instruction_used_when_no_context() -> None:
    out, _ = adapter.render_output(
        "PostToolUse", {"ok": True, "silent": False, "instruction": "record the fix"}
    )
    assert (
        "record the fix" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    )


def test_render_output_stop_uses_system_message() -> None:
    out, _ = adapter.render_output(
        "Stop", {"ok": True, "silent": False, "instruction": "capture learnings"}
    )
    parsed = json.loads(out)
    assert parsed["systemMessage"] == "capture learnings"
    assert "hookSpecificOutput" not in parsed


def test_hook_event_name_authoritative_then_fallback() -> None:
    assert (
        adapter.hook_event_name_of({"hook_event_name": "PreToolUse"}, "pre_edit")
        == "PreToolUse"
    )
    assert adapter.hook_event_name_of({}, "session_start") == "SessionStart"
    assert adapter.hook_event_name_of({}, "stop") == "Stop"
    assert adapter.hook_event_name_of({}, "test_failed") == "PostToolUse"


# --- end-to-end subprocess fail-safety --------------------------------------


def _run_adapter(args, *, stdin: str, env_extra=None):
    env = dict(os.environ)
    env.setdefault("POTPIE_HOOK_DEBUG", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(ADAPTER_PATH), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def _fake_potpie(tmp_path: Path, body: str) -> dict[str, str]:
    """Write a fake `potpie` shell shim on a fresh PATH that echoes `body`."""
    binary = tmp_path / "potpie"
    binary.write_text(f"#!/bin/sh\ncat <<'JSON'\n{body}\nJSON\n", encoding="utf-8")
    binary.chmod(0o755)
    return {"PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"}


def test_subprocess_injects_when_fake_potpie_returns_context(tmp_path: Path) -> None:
    env = _fake_potpie(
        tmp_path,
        json.dumps(
            {
                "ok": True,
                "silent": False,
                "inject_context": "PREF: wrap calls in tenacity",
            }
        ),
    )
    proc = _run_adapter(
        ["--harness", "claude", "--event", "pre_edit"],
        stdin=json.dumps({"session_id": "s1", "tool_input": {"file_path": "src/x.py"}}),
        env_extra=env,
    )
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert "tenacity" in parsed["hookSpecificOutput"]["additionalContext"]


def test_subprocess_silent_when_fake_potpie_silent(tmp_path: Path) -> None:
    env = _fake_potpie(tmp_path, json.dumps({"ok": True, "silent": True}))
    proc = _run_adapter(
        ["--event", "pre_edit"],
        stdin=json.dumps({"tool_input": {"file_path": "src/x.py"}}),
        env_extra=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_subprocess_missing_binary_is_failsafe() -> None:
    proc = _run_adapter(
        ["--event", "pre_edit", "--potpie-bin", "potpie-does-not-exist-zzz"],
        stdin=json.dumps({"tool_input": {"file_path": "src/x.py"}}),
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_subprocess_empty_stdin_is_failsafe(tmp_path: Path) -> None:
    env = _fake_potpie(tmp_path, json.dumps({"ok": True, "inject_context": "x"}))
    proc = _run_adapter(["--event", "session_start"], stdin="", env_extra=env)
    # session_start nudges even with empty payload; should run cleanly.
    assert proc.returncode == 0


def test_subprocess_broken_potpie_output_is_failsafe(tmp_path: Path) -> None:
    env = _fake_potpie(tmp_path, "this is not json {{{")
    proc = _run_adapter(
        ["--event", "pre_edit"],
        stdin=json.dumps({"tool_input": {"file_path": "a.py"}}),
        env_extra=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_subprocess_non_deploy_bash_does_not_call_potpie(tmp_path: Path) -> None:
    # If the adapter wrongly called potpie, the fake would inject; assert it does not.
    env = _fake_potpie(
        tmp_path, json.dumps({"ok": True, "inject_context": "SHOULD-NOT-APPEAR"})
    )
    proc = _run_adapter(
        ["--event", "bash_pre"],
        stdin=json.dumps({"tool_input": {"command": "ls -la"}}),
        env_extra=env,
    )
    assert proc.returncode == 0
    assert "SHOULD-NOT-APPEAR" not in proc.stdout
