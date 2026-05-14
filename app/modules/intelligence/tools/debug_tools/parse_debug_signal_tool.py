"""Parse a pasted log, stack trace, or failed test into a structured debug signal.

Pure-regex extraction — no LLM call. Deterministic and unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Regex bank
# ---------------------------------------------------------------------------

_PYTHON_FRAME = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\S+)'
)
_NODE_FRAME = re.compile(
    r"at\s+(?:(?P<sym>[^\s(]+)\s+)?\((?P<file>[^:)]+):(?P<line>\d+):\d+\)"
)
_NODE_FRAME_NOPARENS = re.compile(
    r"at\s+(?P<sym>[^\s(]+)\s+(?P<file>[^\s:]+):(?P<line>\d+):\d+"
)
_GO_FRAME = re.compile(
    r"(?P<file>[^\s]+\.go):(?P<line>\d+)\s+\+0x"
)
_JAVA_FRAME = re.compile(
    r"at\s+[\w$.]+\((?P<file>[\w$]+\.java):(?P<line>\d+)\)"
)
_GENERIC_PATH_LINE = re.compile(
    r"(?P<file>(?:[\w./\\-]+/)?[\w-]+\.(?:py|ts|js|go|java|rb|rs|cs|cpp|c|h|php|swift))"
    r":(?P<line>\d+)"
)

_ERROR_CLASS_PYTHON = re.compile(r"^(\w+(?:\.\w+)*Error|\w+Exception|\w+Error):\s", re.M)
_ERROR_CLASS_NODE = re.compile(r"^(\w+Error|Error):\s", re.M)
_ERROR_CODE = re.compile(r"\b([A-Z][A-Z0-9_]{2,}(?:_ERROR|_TIMEOUT|_FAILURE|_EXCEPTION))\b")

_REQUEST_ID = re.compile(r"(?:request_id|req_id|requestId)\s*[=:]\s*([^\s,]+)", re.I)
_CORRELATION_ID = re.compile(r"(?:correlation_id|correlationId|x-correlation-id)\s*[=:]\s*([^\s,]+)", re.I)
_TRACE_ID = re.compile(r"(?:trace_id|traceId|x-trace-id)\s*[=:]\s*([^\s,]+)", re.I)

_JEST_EXPECTED = re.compile(r"Expected(?:\s+value)?:\s+(.+?)(?=\n\s*Received|\Z)", re.S)
_JEST_RECEIVED = re.compile(r"Received(?:\s+value)?:\s+(.+?)(?=\n\n|\Z)", re.S)
_PYTEST_EXPECTED = re.compile(r"\+\s*(.+)", re.M)
_PYTEST_RECEIVED = re.compile(r"-\s*(.+)", re.M)
_FAIL_LINE = re.compile(r"(?:FAIL|FAILED|✕|✗)\s+([^\n]+)")
_TEST_PATH = re.compile(r"(?:FAIL|FAILED)\s+([\w./\\-]+(?:test|spec)[\w./\\-]*)", re.I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_frames(frames: List[Dict]) -> List[Dict]:
    seen: set = set()
    out = []
    for f in frames:
        key = (f["file"], f["line"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _parse_frames(text: str) -> List[Dict]:
    frames: List[Dict] = []

    for m in _PYTHON_FRAME.finditer(text):
        frames.append({"file": m.group(1), "line": int(m.group(2)), "symbol": m.group(3)})

    for m in _NODE_FRAME.finditer(text):
        sym = m.group("sym") or "?"
        frames.append({"file": m.group("file"), "line": int(m.group("line")), "symbol": sym})

    for m in _NODE_FRAME_NOPARENS.finditer(text):
        frames.append({"file": m.group("file"), "line": int(m.group("line")), "symbol": m.group("sym")})

    for m in _GO_FRAME.finditer(text):
        frames.append({"file": m.group("file"), "line": int(m.group("line")), "symbol": "?"})

    for m in _JAVA_FRAME.finditer(text):
        frames.append({"file": m.group("file"), "line": int(m.group("line")), "symbol": "?"})

    if not frames:
        for m in _GENERIC_PATH_LINE.finditer(text):
            frames.append({"file": m.group("file"), "line": int(m.group("line")), "symbol": "?"})

    return _dedupe_frames(frames)


def _detect_signal_type(text: str, frames: List[Dict]) -> str:
    lower = text.lower()
    if re.search(r"(FAIL|FAILED|✕|✗|Expected:|Received:|\bfailed\b.*test|\btest.*failed\b)", text):
        return "failed_test"
    if frames:
        if any(kw in lower for kw in ("traceback", "stack trace", "at ", "file \"")):
            return "stack_trace"
        return "pasted_log"
    return "natural_language"


def _extract_error_signature(text: str) -> Optional[str]:
    m = _ERROR_CLASS_PYTHON.search(text) or _ERROR_CLASS_NODE.search(text)
    if m:
        return m.group(1)
    m = _ERROR_CODE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_test_info(text: str) -> Dict[str, Optional[str]]:
    expected = received = test_path = None

    m = _TEST_PATH.search(text)
    if m:
        test_path = m.group(1).strip()

    m = _JEST_EXPECTED.search(text)
    if m:
        expected = m.group(1).strip()[:200]
    m = _JEST_RECEIVED.search(text)
    if m:
        received = m.group(1).strip()[:200]

    return {"expected": expected, "actual": received, "test_path": test_path}


def _format_output(data: Dict[str, Any]) -> str:
    lines = [f"**signal_type**: {data['signal_type']}"]

    if data.get("error_signature"):
        lines.append(f"**error_signature**: {data['error_signature']}")

    if data.get("request_id"):
        lines.append(f"**request_id**: {data['request_id']}")
    if data.get("correlation_id"):
        lines.append(f"**correlation_id**: {data['correlation_id']}")
    if data.get("trace_id"):
        lines.append(f"**trace_id**: {data['trace_id']}")

    frames = data.get("stack_frames", [])
    if frames:
        lines.append(f"\n**stack_frames** ({len(frames)} extracted):")
        for f in frames:
            sym = f.get("symbol") or "?"
            lines.append(f"  - {f['file']}:{f['line']}  in `{sym}`")
    else:
        lines.append("\n**stack_frames**: none extracted")

    ti = data.get("test_info", {})
    if ti.get("test_path"):
        lines.append(f"\n**test_path**: {ti['test_path']}")
    if ti.get("expected"):
        lines.append(f"**expected**: {ti['expected']}")
    if ti.get("actual"):
        lines.append(f"**actual**: {ti['actual']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ParseDebugSignalInput(BaseModel):
    text: str = Field(description="Raw pasted log, stack trace, failed test output, or symptom description")


def parse_debug_signal(input_data: ParseDebugSignalInput) -> str:
    text = input_data.text

    frames = _parse_frames(text)
    signal_type = _detect_signal_type(text, frames)
    error_sig = _extract_error_signature(text)
    test_info = _extract_test_info(text) if signal_type == "failed_test" else {}

    req_m = _REQUEST_ID.search(text)
    corr_m = _CORRELATION_ID.search(text)
    trace_m = _TRACE_ID.search(text)

    data: Dict[str, Any] = {
        "signal_type": signal_type,
        "stack_frames": frames,
        "error_signature": error_sig,
        "request_id": req_m.group(1) if req_m else None,
        "correlation_id": corr_m.group(1) if corr_m else None,
        "trace_id": trace_m.group(1) if trace_m else None,
        "test_info": test_info,
    }

    return _format_output(data)


def create_parse_debug_signal_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=parse_debug_signal,
        name="parse_debug_signal",
        description=(
            "Parse a pasted log, stack trace, failed test output, or symptom description "
            "into a structured signal: signal_type, stack_frames (file/line/symbol), "
            "error_signature, request/correlation/trace ids, and test expected/actual."
        ),
        args_schema=ParseDebugSignalInput,
    )
