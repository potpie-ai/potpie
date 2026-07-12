#!/usr/bin/env python3
# ruff: noqa: S603
"""Potpie nudge adapter — the thin, model-free hook forwarder.

A harness (Claude Code, Codex, Cursor) fires a hook event; this adapter reads the
event payload from stdin, mechanically maps it to a single ``potpie graph nudge``
call, and injects the returned context/instruction back into the session. It owns
**no trigger policy** and makes **no model call**: every decision about *what* to
read, *whether* it is relevant, and *whether* to prompt a write lives inside
``potpie graph nudge`` (which uses only a local embedder). The adapter's only job is
field-forwarding plus the mechanical event-name mapping the harness taxonomy forces
(one ``PostToolUse(Bash)`` event must become either ``test_failed`` or
``test_passed``).

Fail-safe by construction: any internal error, a missing ``potpie`` binary, or an
unparseable payload results in exit code 0 with no output, so a hook problem can
never block or corrupt the user's session. Set ``POTPIE_HOOK_DEBUG=1`` to emit
diagnostics on stderr.

The pure functions below (event mapping, command classification, argv building,
output rendering) carry the logic and are unit-tested directly; ``main`` is the thin
stdin/subprocess/stdout shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

# Nudge events understood by `potpie graph nudge` (must match domain.nudge.NudgeEvent).
NUDGE_EVENTS = frozenset(
    {"session_start", "pre_edit", "pre_deploy", "test_failed", "test_passed", "stop"}
)


def canonical_nudge_event(value: str | None) -> str | None:
    if not value:
        return None
    event = str(value).strip().replace("-", "_")
    return event if event in NUDGE_EVENTS else None


# Map a nudge event back to the harness hook-event name used in the output envelope.
_CLAUDE_EVENT_FOR_NUDGE = {
    "session_start": "SessionStart",
    "pre_edit": "PreToolUse",
    "pre_deploy": "PreToolUse",
    "test_failed": "PostToolUse",
    "test_passed": "PostToolUse",
    "stop": "Stop",
}

# Substrings that mark a shell command as a deploy / infra-changing action.
_DEPLOY_MARKERS = (
    "kubectl apply",
    "kubectl rollout",
    "kubectl delete",
    "helm upgrade",
    "helm install",
    "terraform apply",
    "terraform destroy",
    "docker push",
    "docker compose up",
    "docker-compose up",
    "serverless deploy",
    "sls deploy",
    "pulumi up",
    "aws deploy",
    "gcloud run deploy",
    "gcloud app deploy",
    "flyctl deploy",
    "fly deploy",
    "cdk deploy",
    "ansible-playbook",
)

# Substrings that mark a shell command as a test / check run.
_TEST_MARKERS = (
    "pytest",
    "py.test",
    "unittest",
    "nox",
    "tox",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "jest",
    "vitest",
    "mocha",
    "go test",
    "cargo test",
    "gradle test",
    "mvn test",
    "rspec",
    "phpunit",
    "ctest",
    "make test",
    "make check",
)

# Classify a finished test run when no exit code is given (the live path on Claude
# Code, whose Bash tool_response carries no exit code). Count-aware first: a numeric
# "N failed/failing/errors" with N==0 is a PASS, not a FAIL — so a green run that
# prints "0 failed" (cargo, jest) is not misread. Bare words like "error" or
# "failed" are deliberately NOT failure markers, because they appear constantly in
# green output (test names, package paths, log lines).
_FAIL_COUNT_RE = re.compile(
    r"(\d+)\s+(?:failed|failing|failures|errors?|broken)\b", re.IGNORECASE
)
_PASS_COUNT_RE = re.compile(r"(\d+)\s+(?:passed|passing)\b", re.IGNORECASE)
_GO_OK_RE = re.compile(r"(?m)^ok\s")  # go test success line: "ok\tpkg\t0.01s"

# Anchored failure markers — only consulted when no numeric failure count is found.
_FAIL_MARKERS = (
    "assertionerror",
    "traceback (most recent call last)",
    "panic:",
    "fatal:",
    "build failed",
    "tests failed",
    "test failed",
    "--- fail",  # go test -v
    "=== fail",
    " fail:",
    "not ok",  # TAP
    "✗",
    "✘",
)
# Anchored success markers.
_PASS_MARKERS = (
    "test result: ok",
    "build succeeded",
    "all tests passed",
    "tests passed",
    " passing",  # mocha "N passing"
    "passed",  # pytest "N passed"
    "✓",
    "✔",
)


def _debug(message: str) -> None:
    if os.environ.get("POTPIE_HOOK_DEBUG"):
        sys.stderr.write(f"[potpie-nudge] {message}\n")


# --- payload accessors (harness-tolerant) -----------------------------------


def _first(payload: dict[str, Any], *paths: str) -> Any:
    """Return the first present value among dotted key paths (e.g. 'tool_input.command')."""
    for path in paths:
        node: Any = payload
        ok = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and node not in (None, ""):
            return node
    return None


def session_id_of(payload: dict[str, Any]) -> str:
    value = _first(payload, "session_id", "sessionId", "conversation_id", "session.id")
    if value:
        return str(value)
    env = os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("POTPIE_SESSION_ID")
    return env or "default"


def tool_name_of(payload: dict[str, Any]) -> str:
    return str(_first(payload, "tool_name", "toolName", "tool.name") or "")


def file_path_of(payload: dict[str, Any]) -> str | None:
    value = _first(
        payload,
        "tool_input.file_path",
        "tool_input.path",
        "toolInput.file_path",
        "params.file_path",
        "file_path",
        "path",
    )
    return str(value) if value else None


def command_of(payload: dict[str, Any]) -> str | None:
    value = _first(
        payload,
        "tool_input.command",
        "toolInput.command",
        "params.command",
        "command",
    )
    return str(value) if value else None


def hook_event_name_of(payload: dict[str, Any], nudge_event: str) -> str:
    """Authoritative harness event name for the output envelope."""
    value = _first(payload, "hook_event_name", "hookEventName", "event")
    if isinstance(value, str) and value in {
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    }:
        return value
    return _CLAUDE_EVENT_FOR_NUDGE.get(nudge_event, "PreToolUse")


# --- mechanical classification ----------------------------------------------


def is_deploy_command(command: str | None) -> bool:
    if not command:
        return False
    low = command.lower()
    return any(marker in low for marker in _DEPLOY_MARKERS)


def is_test_command(command: str | None) -> bool:
    if not command:
        return False
    low = command.lower()
    return any(marker in low for marker in _TEST_MARKERS)


def _exit_code_of(tool_response: Any) -> int | None:
    if not isinstance(tool_response, dict):
        return None
    for key in ("exit_code", "exitCode", "returncode", "return_code", "code", "status"):
        if key in tool_response:
            try:
                return int(tool_response[key])
            except (TypeError, ValueError):
                continue
    return None


def _response_text(tool_response: Any) -> str:
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        parts = [
            str(tool_response.get(k, ""))
            for k in ("stdout", "stderr", "output", "content", "result", "message")
        ]
        return "\n".join(p for p in parts if p)
    return ""


def test_outcome(tool_response: Any) -> str | None:
    """Return 'pass' | 'fail' | None (ambiguous → skip) for a finished test run.

    Mechanical only. Decision order: an explicit exit code wins; then a numeric
    failure count (``N failed`` with N==0 → pass, the common green-run shape); then
    anchored markers. Bare words ("error", "failed") are never enough on their own —
    they appear in green output — so an ambiguous run returns None (stay silent)
    rather than firing a false ``test_failed`` nudge.
    """
    if isinstance(tool_response, dict) and tool_response.get("interrupted"):
        return None
    code = _exit_code_of(tool_response)
    if code is not None:
        return "pass" if code == 0 else "fail"
    if isinstance(tool_response, dict) and tool_response.get("is_error") is True:
        return "fail"
    text = _response_text(tool_response)
    if not text:
        return None
    low = text.lower()

    # Numeric failure/error counts are the strongest text signal: zero ⇒ pass.
    fail_counts = [int(n) for n in _FAIL_COUNT_RE.findall(low)]
    if fail_counts:
        return "fail" if any(n > 0 for n in fail_counts) else "pass"

    # No numeric count — fall back to anchored markers.
    has_fail = any(marker in low for marker in _FAIL_MARKERS)
    has_pass = (
        any(marker in low for marker in _PASS_MARKERS)
        or _PASS_COUNT_RE.search(low) is not None
        or _GO_OK_RE.search(text) is not None
    )
    if has_fail and not has_pass:
        return "fail"
    if has_pass and not has_fail:
        return "pass"
    return None


def _failure_symptom(command: str | None, tool_response: Any) -> str | None:
    """Extract a compact symptom query from a failed run (for prior-bug matching)."""
    text = _response_text(tool_response)
    best = None
    for line in text.splitlines():
        low = line.strip().lower()
        if not low:
            continue
        if any(
            sig in low
            for sig in ("error", "assert", "failed", "exception", "traceback")
        ):
            best = line.strip()
            break
    parts = [p for p in (command, best) if p]
    if not parts:
        return None
    return " — ".join(parts)[:300]


def resolve_nudge_event(
    hint: str, payload: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    """Map a harness hook-event hint + payload to a nudge event and its fields.

    Returns ``(None, {})`` when the event should not nudge (e.g. a non-deploy Bash
    pre-hook, or an ambiguous test outcome) — the adapter then stays silent.
    """
    hint = (hint or "").strip().lower()
    if hint == "session_start":
        return "session_start", {}
    if hint == "stop":
        return "stop", {}
    if hint == "pre_edit":
        return "pre_edit", {"path": file_path_of(payload)}
    if hint == "bash_pre":
        command = command_of(payload)
        if is_deploy_command(command):
            return "pre_deploy", {"query": command}
        return None, {}
    if hint == "bash_post":
        command = command_of(payload)
        if not is_test_command(command):
            return None, {}
        tool_response = _first(payload, "tool_response", "toolResponse", "result")
        outcome = test_outcome(tool_response)
        if outcome == "fail":
            return "test_failed", {"query": _failure_symptom(command, tool_response)}
        if outcome == "pass":
            return "test_passed", {"query": command}
        return None, {}
    # Direct nudge events may be passed straight through (Codex/Cursor wiring).
    direct_event = canonical_nudge_event(hint)
    if direct_event:
        return direct_event, {
            "path": file_path_of(payload),
            "query": command_of(payload),
        }
    return None, {}


# --- CLI argv + output rendering --------------------------------------------


def build_argv(
    potpie_bin: str,
    nudge_event: str,
    session: str,
    *,
    path: str | None = None,
    query: str | None = None,
    pot: str | None = None,
    limit: int | None = None,
) -> list[str]:
    argv = [
        potpie_bin,
        "--json",
        "graph",
        "nudge",
        "--event",
        nudge_event,
        "--session",
        session,
    ]
    if path:
        argv += ["--path", path]
    if query:
        argv += ["--query", query]
    if pot:
        argv += ["--pot", pot]
    if limit is not None:
        argv += ["--limit", str(limit)]
    return argv


def render_output(claude_event: str, nudge_result: Any) -> tuple[str, int]:
    """Shape a nudge result into harness hook output. Always exit 0 (never block)."""
    if not isinstance(nudge_result, dict):
        return "", 0
    ok = bool(nudge_result.get("ok"))
    if isinstance(nudge_result.get("result"), dict):
        # Graph V2 workbench wraps command bodies under result. Keep accepting
        # the old flat V1.5 nudge shape so installed hooks do not need a lockstep
        # CLI upgrade.
        nudge_result = nudge_result["result"]
    else:
        ok = bool(nudge_result.get("ok"))
    if not ok or nudge_result.get("silent"):
        return "", 0
    text = nudge_result.get("inject_context") or nudge_result.get("instruction")
    if not text:
        return "", 0
    if claude_event == "Stop":
        # additionalContext is not honored at Stop; surface as a user-visible note.
        return json.dumps({"systemMessage": str(text)}), 0
    envelope = {
        "hookSpecificOutput": {
            "hookEventName": claude_event,
            "additionalContext": str(text),
        }
    }
    return json.dumps(envelope), 0


# --- main (thin shell) ------------------------------------------------------


def _read_stdin_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except Exception:  # noqa: BLE001 - stdin may be closed; treat as empty
        return {}
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        add_help=True, description="Potpie nudge hook adapter."
    )
    parser.add_argument("--harness", default="claude", help="claude|codex|cursor")
    parser.add_argument(
        "--event",
        required=True,
        help="hook hint: session_start|pre_edit|bash_pre|bash_post|stop "
        "(or a direct nudge event)",
    )
    parser.add_argument("--pot", default=os.environ.get("POTPIE_POT"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--potpie-bin", default=os.environ.get("POTPIE_BIN", "potpie"))
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # argparse already wrote usage; never propagate a nonzero exit to the harness.
        return 0

    try:
        payload = _read_stdin_payload()
        nudge_event, fields = resolve_nudge_event(args.event, payload)
        if nudge_event is None:
            _debug(f"no nudge for hint={args.event!r}; staying silent")
            return 0

        binary = shutil.which(args.potpie_bin) or args.potpie_bin
        if shutil.which(args.potpie_bin) is None and not os.path.exists(binary):
            _debug(f"potpie binary {args.potpie_bin!r} not found; staying silent")
            return 0

        session = session_id_of(payload)
        cmd = build_argv(
            binary,
            nudge_event,
            session,
            path=fields.get("path"),
            query=fields.get("query"),
            pot=args.pot,
            limit=args.limit,
        )
        _debug(f"running: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(os.environ.get("POTPIE_HOOK_TIMEOUT", "15")),
            check=False,
        )
        if proc.returncode != 0:
            _debug(f"nudge exited {proc.returncode}: {proc.stderr.strip()[:200]}")
            return 0
        try:
            result = json.loads(proc.stdout or "{}")
        except (ValueError, TypeError):
            _debug(f"unparseable nudge output: {proc.stdout[:200]!r}")
            return 0

        claude_event = hook_event_name_of(payload, nudge_event)
        out, code = render_output(claude_event, result)
        if out:
            sys.stdout.write(out)
        return code
    except subprocess.TimeoutExpired:
        _debug("nudge timed out; staying silent")
        return 0
    except Exception as exc:  # noqa: BLE001 - a hook must never break the session
        _debug(f"unexpected error: {exc!r}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
