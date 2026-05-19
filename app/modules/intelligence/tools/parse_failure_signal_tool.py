"""parse_failure_signal — pure-parsing tool for the debug agent.

Takes a raw failure text (log, stack trace, failed-test output, or natural-language
symptom) and returns a structured representation: classification, extracted stack
frames, error type, and a short signature for dedup/grouping.

No I/O, no network, no LLM calls.  Entirely deterministic regex-based extraction.
"""

import re
from typing import Dict, Any, List, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

InputTypeHint = Optional[Literal["pasted_log", "nl_symptom", "failed_test"]]
Classification = Literal["pasted_log", "nl_symptom", "failed_test"]
Language = Literal["python", "javascript", "typescript", "go", "unknown"]


class StackFrame(BaseModel):
    file: str
    line: Optional[int] = None
    symbol: Optional[str] = None
    language: Language = "unknown"


class ParseFailureSignalResult(BaseModel):
    signature: str
    classification: Classification
    stack_frames: List[StackFrame]
    error_type: Optional[str] = None
    raw_excerpt: str


class ParseFailureSignalInput(BaseModel):
    raw_text: str = Field(description="Pasted log, stack trace, or failed-test output")
    input_type_hint: InputTypeHint = Field(
        default=None,
        description='Optional classification hint: "pasted_log", "nl_symptom", or "failed_test"',
    )


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _language_from_file(path: str) -> Language:
    path_lower = path.lower()
    if path_lower.endswith(".py"):
        return "python"
    if path_lower.endswith(".ts") or path_lower.endswith(".tsx"):
        return "typescript"
    if path_lower.endswith(".js") or path_lower.endswith(".jsx") or path_lower.endswith(".mjs"):
        return "javascript"
    if path_lower.endswith(".go"):
        return "go"
    return "unknown"


# ---------------------------------------------------------------------------
# Extractor: Python traceback
# ---------------------------------------------------------------------------
# Shape:
#   File "path/to/file.py", line 42, in function_name
#   <source line>
# Error line at end: ErrorClass: message   OR   module.ErrorClass: message
_PY_FRAME_RE = re.compile(
    r'^\s{2}File "(?P<file>[^"]+)",\s*line\s+(?P<line>\d+),\s*in\s+(?P<symbol>\S+)',
    re.MULTILINE,
)
_PY_ERROR_RE = re.compile(
    r'^(?P<error>[\w.]+(?:Error|Exception|Warning|Fault|Timeout|Interrupt|Exit|Stop|KeyboardInterrupt))[:\s]',
    re.MULTILINE,
)


def _extract_python(text: str) -> Optional[List[StackFrame]]:
    frames = []
    for m in _PY_FRAME_RE.finditer(text):
        frames.append(
            StackFrame(
                file=m.group("file"),
                line=int(m.group("line")),
                symbol=m.group("symbol"),
                language=_language_from_file(m.group("file")),
            )
        )
    return frames if frames else None


def _python_error_type(text: str) -> Optional[str]:
    m = _PY_ERROR_RE.search(text)
    if m:
        name = m.group("error")
        # Return the leaf class name (after last dot)
        return name.split(".")[-1]
    return None


# ---------------------------------------------------------------------------
# Extractor: pytest failure
# ---------------------------------------------------------------------------
# Shape:
#   FAILED tests/foo.py::test_bar - AssertionError: ...
#   OR inline assertion lines inside the boxed traceback
_PYTEST_FAILED_RE = re.compile(
    r'^FAILED\s+(?P<path>[\w/\\.\-]+::[\w.\-]+)\s+-\s+(?P<error>[\w.]+(?:Error|Exception|Warning|Assertion)[^:\n]*)',
    re.MULTILINE,
)
_PYTEST_ASSERT_RE = re.compile(
    r'^E\s+(?P<error>[\w]+(?:Error|Exception|Assertion))[:\s]',
    re.MULTILINE,
)


def _extract_pytest(text: str) -> Optional[List[StackFrame]]:
    """Return frames if this looks like a pytest failure output."""
    failed_match = _PYTEST_FAILED_RE.search(text)
    assert_match = _PYTEST_ASSERT_RE.search(text)
    if not failed_match and not assert_match:
        return None

    # Also pull Python frames from the traceback section
    frames = _extract_python(text) or []

    # If no Python frames but we found a FAILED line, synthesise one frame
    if not frames and failed_match:
        path_part = failed_match.group("path")
        file_part = path_part.split("::")[0]
        frames.append(
            StackFrame(
                file=file_part,
                line=None,
                symbol=None,
                language=_language_from_file(file_part),
            )
        )
    return frames if frames else None


def _pytest_error_type(text: str) -> Optional[str]:
    m = _PYTEST_FAILED_RE.search(text)
    if m:
        raw = m.group("error").strip()
        # strip trailing message after ':'
        cls = raw.split(":")[0].strip()
        return cls.split(".")[-1]
    m2 = _PYTEST_ASSERT_RE.search(text)
    if m2:
        return m2.group("error").strip()
    return None


# ---------------------------------------------------------------------------
# Extractor: Node / TypeScript / JavaScript stack
# ---------------------------------------------------------------------------
# Shape:
#   at functionName (path/to/file.ts:42:13)
#   at path/to/file.js:88:7
_NODE_AT_FULL_RE = re.compile(
    r'^\s+at\s+(?P<symbol>[\w.<>$\[\] ]+?)\s+\((?P<file>[^)]+):(?P<line>\d+):\d+\)',
    re.MULTILINE,
)
_NODE_AT_BARE_RE = re.compile(
    r'^\s+at\s+(?P<file>(?!\s)[\w./\\@\-]+\.[jt]sx?):(?P<line>\d+):\d+',
    re.MULTILINE,
)
# Error class on the first line: "ErrorClass: message"
_NODE_ERROR_RE = re.compile(
    r'^(?P<error>[\w]+(?:Error|Exception|Timeout|Fault))[:\s]',
    re.MULTILINE,
)


def _extract_node(text: str) -> Optional[List[StackFrame]]:
    frames = []
    # Combine both patterns in order of appearance
    hits: List[tuple] = []  # (match_start, frame)

    for m in _NODE_AT_FULL_RE.finditer(text):
        raw_file = m.group("file")
        # file may have leading "file://" or similar — strip to path
        file_path = _FILE_URI_RE.sub('', raw_file)
        # strip column (already in regex)
        hits.append(
            (
                m.start(),
                StackFrame(
                    file=file_path,
                    line=int(m.group("line")),
                    symbol=m.group("symbol").strip(),
                    language=_language_from_file(file_path),
                ),
            )
        )

    for m in _NODE_AT_BARE_RE.finditer(text):
        file_path = m.group("file")
        hits.append(
            (
                m.start(),
                StackFrame(
                    file=file_path,
                    line=int(m.group("line")),
                    symbol=None,
                    language=_language_from_file(file_path),
                ),
            )
        )

    # Sort by position and deduplicate (full match wins over bare if same position)
    hits.sort(key=lambda x: x[0])
    seen_pos: set = set()
    for pos, frame in hits:
        if pos not in seen_pos:
            frames.append(frame)
            seen_pos.add(pos)

    return frames if frames else None


def _node_error_type(text: str) -> Optional[str]:
    m = _NODE_ERROR_RE.search(text)
    return m.group("error") if m else None


# ---------------------------------------------------------------------------
# Extractor: Jest / Mocha failure
# ---------------------------------------------------------------------------
# Shape:
#   FAIL tests/foo.test.ts
#   ● Test Suite > test name
#   Expected: ...  Received: ...
_JEST_FAIL_RE = re.compile(r'^[\s]*(FAIL|FAILED)\s+[\w/\\.\-]+\.(?:test|spec)\.[jt]sx?', re.MULTILINE)
_JEST_MOCHA_MARKER_RE = re.compile(r'^[\s]*●\s+', re.MULTILINE)


def _extract_jest(text: str) -> Optional[List[StackFrame]]:
    """Return frames if text looks like Jest/Mocha output."""
    fail_match = _JEST_FAIL_RE.search(text)
    mocha_match = _JEST_MOCHA_MARKER_RE.search(text)
    if not fail_match and not mocha_match:
        return None

    # Pull any Node-style "at ..." frames from the traceback section
    frames = _extract_node(text) or []

    # Synthesise a frame from the FAIL file path if no other frames
    if not frames and fail_match:
        line_text = fail_match.group(0).strip()
        # The file path is after FAIL/FAILED token
        parts = line_text.split(None, 1)
        if len(parts) == 2:
            file_part = parts[1].strip()
            frames.append(
                StackFrame(
                    file=file_part,
                    line=None,
                    symbol=None,
                    language=_language_from_file(file_part),
                )
            )
    return frames if frames else None


def _jest_error_type(text: str) -> Optional[str]:
    # Jest usually shows "Expected: N  Received: M" without explicit error class.
    # Check for explicit thrown errors in "at" lines.
    return _node_error_type(text)


# ---------------------------------------------------------------------------
# Extractor: Go goroutine dump
# ---------------------------------------------------------------------------
# Shape:
#   goroutine 1 [running]:
#   main.foo(0xc...)
#         /path/to/file.go:42 +0x...
_GO_GOROUTINE_RE = re.compile(r'goroutine\s+\d+\s+\[', re.MULTILINE)
_GO_SYMBOL_RE = re.compile(r'^(?P<symbol>[\w/.]+)\(', re.MULTILINE)
_GO_FILE_RE = re.compile(r'^\t(?P<file>[^\s:]+\.go):(?P<line>\d+)', re.MULTILINE)

# Promoted inline patterns (Minor 2)
_FILE_URI_RE = re.compile(r'^file://')
_PATH_PREFIX_RE = re.compile(r'^.*?(?:src/|app/|tests/|lib/)')


def _extract_go(text: str) -> Optional[List[StackFrame]]:
    if not _GO_GOROUTINE_RE.search(text):
        return None

    frames = []
    # Pair symbol lines with their following file lines
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        sym_m = _GO_SYMBOL_RE.match(line)
        if sym_m and i + 1 < len(lines):
            file_m = _GO_FILE_RE.match(lines[i + 1])
            if file_m:
                symbol = sym_m.group("symbol")
                file_path = file_m.group("file")
                line_no = int(file_m.group("line"))
                frames.append(
                    StackFrame(
                        file=file_path,
                        line=line_no,
                        symbol=symbol,
                        language="go",
                    )
                )
                i += 2
                continue
        i += 1

    return frames if frames else None


# ---------------------------------------------------------------------------
# Detection logic with precedence
# ---------------------------------------------------------------------------

def _is_pytest(text: str) -> bool:
    return bool(
        _PYTEST_FAILED_RE.search(text)
        or (_PY_FRAME_RE.search(text) and _PYTEST_ASSERT_RE.search(text))
    )


def _is_python_traceback(text: str) -> bool:
    return bool(_PY_FRAME_RE.search(text))


def _is_jest(text: str) -> bool:
    fail_line = bool(_JEST_FAIL_RE.search(text))
    mocha_marker = bool(_JEST_MOCHA_MARKER_RE.search(text))
    if mocha_marker and not fail_line:
        return bool(_NODE_AT_FULL_RE.search(text) or _NODE_AT_BARE_RE.search(text))
    return fail_line or mocha_marker


def _is_node(text: str) -> bool:
    return bool(_NODE_AT_FULL_RE.search(text) or _NODE_AT_BARE_RE.search(text))


def _is_go(text: str) -> bool:
    return bool(_GO_GOROUTINE_RE.search(text))


# ---------------------------------------------------------------------------
# Signature builder
# ---------------------------------------------------------------------------

def _build_signature(
    text: str,
    classification: Classification,
    frames: List[StackFrame],
    error_type: Optional[str],
) -> str:
    if classification == "nl_symptom" or not frames:
        # Use first ~60 chars of raw text
        cleaned = " ".join(text.split())
        return cleaned[:60]

    # Build from error_type + first meaningful frame
    parts = []
    if error_type:
        parts.append(error_type)
    if frames:
        first = frames[0]
        # Strip leading path components to keep it short
        short_file = _PATH_PREFIX_RE.sub('', first.file)
        short_file = short_file.rstrip("/").replace("/", ".")
        if short_file.endswith(".py"):
            short_file = short_file[:-3]
        elif short_file.endswith(".tsx"):
            short_file = short_file[:-4]
        elif short_file.endswith(".ts"):
            short_file = short_file[:-3]
        elif short_file.endswith(".jsx"):
            short_file = short_file[:-4]
        elif short_file.endswith(".js"):
            short_file = short_file[:-3]
        elif short_file.endswith(".go"):
            short_file = short_file[:-3]
        if first.symbol:
            parts.append(f"{short_file}.{first.symbol}")
        else:
            parts.append(short_file)
    sig = ".".join(parts) if parts else " ".join(text.split())[:60]
    return sig[:80]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def parse_failure_signal(
    raw_text: str,
    input_type_hint: InputTypeHint = None,
) -> Dict[str, Any]:
    """Parse a raw failure signal and return structured debug information.

    Pure parsing — no I/O, no network, no LLM calls.
    """
    text = raw_text or ""
    raw_excerpt = text[:300]

    # --- Auto-detection (ordered by specificity: most specific first) ---
    # pytest > python_traceback > jest/mocha > node > go > nl_symptom

    detected_classification: Classification = "nl_symptom"
    frames: Optional[List[StackFrame]] = None
    error_type: Optional[str] = None

    if _is_pytest(text):
        detected_classification = "failed_test"
        frames = _extract_pytest(text)
        error_type = _pytest_error_type(text)
    elif _is_python_traceback(text):
        detected_classification = "pasted_log"
        frames = _extract_python(text)
        error_type = _python_error_type(text)
    elif _is_jest(text):
        detected_classification = "failed_test"
        frames = _extract_jest(text)
        error_type = _jest_error_type(text)
    elif _is_node(text):
        detected_classification = "pasted_log"
        frames = _extract_node(text)
        error_type = _node_error_type(text)
    elif _is_go(text):
        detected_classification = "pasted_log"
        frames = _extract_go(text)
        error_type = None
    else:
        detected_classification = "nl_symptom"
        frames = []
        error_type = None

    # --- Honor hint ---
    # When a hint is provided, override the classification.
    # The hint does NOT change frame extraction (we already extracted from the content);
    # it only affects the final classification label.
    if input_type_hint is not None:
        final_classification: Classification = input_type_hint
    else:
        final_classification = detected_classification

    # If detected as nl_symptom (no frames), keep frames empty regardless of hint
    safe_frames: List[StackFrame] = frames or []

    # --- Signature ---
    signature = _build_signature(text, final_classification, safe_frames, error_type)

    result = ParseFailureSignalResult(
        signature=signature,
        classification=final_classification,
        stack_frames=safe_frames,
        error_type=error_type,
        raw_excerpt=raw_excerpt,
    )

    # Return as dict so existing StructuredTool plumbing serializes it cleanly
    return result.model_dump()


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory
# ---------------------------------------------------------------------------

def parse_failure_signal_tool() -> StructuredTool:
    """Create the parse_failure_signal StructuredTool.

    No db/user_id required — this tool is pure parsing with no I/O.
    """
    return StructuredTool.from_function(
        func=parse_failure_signal,
        name="parse_failure_signal",
        description=(
            "Parse a pasted log, stack trace, or failed-test output and return a "
            "structured representation: classification (pasted_log | failed_test | nl_symptom), "
            "extracted stack frames (file, line, symbol, language), error type, and a short "
            "signature for grouping. Pure parsing — no network, no LLM. "
            "Use this as the first step in a debugging session to extract candidate source "
            "locations before forming hypotheses."
        ),
        args_schema=ParseFailureSignalInput,
    )
