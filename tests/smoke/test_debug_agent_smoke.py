#!/usr/bin/env python3
"""Smoke test — DebugAgent hypothesis creation via real LLM.

Runs the actual debug_task_prompt against a real LLM and measures whether the
agent follows the hypothesis-driven debugging protocol.

Usage
-----
Standalone:
    python tests/smoke/test_debug_agent_smoke.py [--model MODEL]

Pytest:
    pytest tests/smoke/test_debug_agent_smoke.py -s -v -m smoke

Required env var:
    ANTHROPIC_API_KEY   (preferred — claude supports tool calling reliably)
  or
    OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY

Optional env vars:
    SMOKE_MODEL   model name override (default: claude-haiku-4-5-20251001)
    SMOKE_TIMEOUT maximum seconds to wait for the agent (default: 120)

What is checked
---------------
The checks mirror the OUTPUT CONTRACT in debug_task_prompt:

  phase1_parse_failure_signal_called  — Phase 1: REQUIRED tool call
  phase2_debugger_status_emitted      — Phase 2: **Debugger:** line emitted
  phase4_hypothesis_markdown_emitted  — Phase 4: ## Hypothesis N: block present
  phase4_record_hypothesis_called     — Phase 4: persistence tool called
  phase4_hypothesis_store_populated   — Phase 4: store has ≥1 record
  phase4_card_terminator_present      — Phase 4: '---' card terminator present
  forbidden_output_absent             — FORBIDDEN OUTPUT: no standalone conclusions
  phase4_pause_question_present       — MULTI-TURN: response ends with pause question
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — works whether called as a module or directly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Env vars that app modules read at import time
# ---------------------------------------------------------------------------
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Stub heavy ML packages pulled in transitively (same pattern as unit tests)
# ---------------------------------------------------------------------------
def _stub(name: str, **attrs: Any) -> types.ModuleType:
    mod = sys.modules.get(name) or types.ModuleType(name)
    sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod

_t = _stub("torch")
_stub("torch.nn", functional=MagicMock())
_stub("torch.nn.functional")
_stub("torch.nn.modules")
_stub("torch._jit_internal")
_stub("torch._sources")
_stub("torch._VF")
_t._VF = MagicMock()
_t.functional = MagicMock()
_t.nn = sys.modules["torch.nn"]
_st = _stub("sentence_transformers")
_st.SentenceTransformer = MagicMock(name="SentenceTransformer")

# ---------------------------------------------------------------------------
# App imports (after stubs)
# ---------------------------------------------------------------------------
from pydantic_ai import Agent, Tool  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402

from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent_prompt import (  # noqa: E402
    debug_task_prompt,
)
from app.modules.intelligence.tools.hypothesis_state_tool import (  # noqa: E402
    HypothesisStore,
    _hypothesis_store_ctx,
    create_hypothesis_state_tools,
)
from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (  # noqa: E402
    wrap_structured_tools,
)

# ---------------------------------------------------------------------------
# Performance contract — assertions written first (TDD RED)
# ---------------------------------------------------------------------------

# Output sections that violate the FORBIDDEN OUTPUT clause
_FORBIDDEN_SECTIONS = [
    "## Root Cause Analysis",
    "## Analysis Summary",
    "## Final Summary",
    "## Vulnerability Overview",
    "## Security Impact",
    "## Recommended Fix",
    "## Bug Analysis",
    "## Detailed Analysis",
    "## Refined Analysis",
    "## Conclusion",
]

# Phrases the model must include at Phase 4 exit to satisfy the MULTI-TURN STOP
_PAUSE_PHRASES = [
    "which hypothesis",
    "Which hypothesis",
    "H1",
    "H2",
    "all of them",
    "start with",
    "pick",
    "validate",
]


def check_protocol_compliance(
    full_text: str,
    tool_calls: list[str],
    hypotheses: list[dict],
) -> dict[str, bool]:
    """Evaluate agent output against every requirement in debug_task_prompt.

    Arguments
    ---------
    full_text  : ALL text emitted across every LLM turn (not just the final turn).
                 The **Debugger:** line is emitted in an intermediate turn; checking
                 only result.output would miss it.
    tool_calls : Ordered list of tool names called during the run.
    hypotheses : Snapshot of the hypothesis store captured INSIDE the async run
                 (the ContextVar goes out of scope after asyncio.run() returns).

    Returns
    -------
    Mapping of requirement_name → passed.  All must be True for the smoke test
    to pass.
    """
    return {
        # --- Phase 1 (REQUIRED — never skip) ---
        "phase1_parse_failure_signal_called": (
            "parse_failure_signal" in tool_calls
        ),
        # --- Phase 2 (Debugger status line) ---
        "phase2_debugger_status_emitted": (
            "**Debugger:**" in full_text
        ),
        # --- Phase 4 (Hypothesis generation) ---
        "phase4_hypothesis_markdown_emitted": (
            "## Hypothesis" in full_text
        ),
        "phase4_record_hypothesis_called": (
            "record_hypothesis" in tool_calls
        ),
        "phase4_hypothesis_store_populated": (
            len(hypotheses) >= 1
        ),
        "phase4_card_terminator_present": (
            "---" in full_text
        ),
        # --- FORBIDDEN OUTPUT contract ---
        "forbidden_output_absent": (
            not any(s in full_text for s in _FORBIDDEN_SECTIONS)
        ),
        # --- MULTI-TURN FLOW (pause after Phase 4) ---
        "phase4_pause_question_present": (
            any(p in full_text for p in _PAUSE_PHRASES)
        ),
    }


# ---------------------------------------------------------------------------
# Debugging signal — real-world ASan heap-buffer-overflow from valkey-io/valkey
# This is the same signal used in the two production traces that failed.
# ---------------------------------------------------------------------------
DEBUGGING_SIGNAL = """\
AddressSanitizer: heap-buffer-overflow on address 0x602000000631 READ of size 1
    #0 0x555b4a1c2340 in zipmapNext zipmap.c:240
    #1 0x555b4a1c3b20 in rdbLoadObject rdb.c:2408
    #2 0x555b4a1c4100 in restoreCommand t_string.c:462

Internal error in RDB reading offset 0, function at rdb.c:2408 -> Hash zipmap \
with dup elements, or big length (0)
"""

# ---------------------------------------------------------------------------
# Mock tools — controlled responses for all infrastructure tools the agent
# may call, so the smoke test runs without a real VS Code / sandbox / DB.
# DAP tools return error_type="no_tunnel" so the agent takes the Case C path.
# ---------------------------------------------------------------------------
def _mock(name: str, description: str, returns: Any) -> Tool:
    async def _fn(**kwargs: Any) -> Any:  # noqa: ANN401
        return returns
    return Tool(name=name, description=description, function=_fn)


_MOCK_TOOLS: list[Tool] = [
    _mock(
        "parse_failure_signal",
        "Parse and structure a raw failure signal.",
        {
            "classification": "pasted_log",
            "signature": "heap-buffer-overflow in zipmapNext",
            "error_type": "HeapBufferOverflow",
            "stack_frames": [
                {"file": "src/zipmap.c", "line": 240, "symbol": "zipmapNext"},
                {"file": "src/rdb.c", "line": 2408, "symbol": "rdbLoadObject"},
            ],
            "raw_excerpt": (
                "AddressSanitizer: heap-buffer-overflow READ of size 1\n"
                "#0 zipmapNext zipmap.c:240\n"
                "#1 rdbLoadObject rdb.c:2408\n"
            ),
        },
    ),
    _mock(
        "get_workspace_debug_context",
        "Return debug capabilities: launch configs, adapters, recent git changes.",
        {
            "available": False,
            "launch_configs": [],
            "inferred_commands": [],
            "recent_changes": [],
            "related_tests": [],
            "message": "No workspace attached in smoke-test mode.",
        },
    ),
    _mock(
        "query_context_graph",
        "Query the context graph for relevant code entities.",
        {"available": False, "results": [], "message": "Context graph not wired."},
    ),
    _mock(
        "ask_knowledge_graph_queries",
        "Semantic search over the embedded knowledge graph.",
        {
            "results": [
                {
                    "file": "src/zipmap.c",
                    "symbol": "zipmapValidateIntegrity",
                    "snippet": (
                        "int zipmapValidateIntegrity(unsigned char *p, "
                        "size_t len, int deep)"
                    ),
                    "score": 0.93,
                },
                {
                    "file": "src/zipmap.c",
                    "symbol": "zipmapNext",
                    "snippet": (
                        "unsigned char *zipmapNext(unsigned char *zm, "
                        "unsigned char **key, unsigned int *klen, "
                        "unsigned char **value, unsigned int *vlen)"
                    ),
                    "score": 0.88,
                },
                {
                    "file": "src/zipmap.c",
                    "symbol": "zipmapDecodeLength",
                    "snippet": (
                        "static unsigned int zipmapDecodeLength(unsigned char *p)"
                    ),
                    "score": 0.81,
                },
            ]
        },
    ),
    _mock(
        "search_text",
        "Ripgrep-style text search over the repository.",
        {
            "results": [
                {"file": "src/zipmap.c", "line": 196, "text": "l = zipmapDecodeLength(p);"},
                {
                    "file": "src/zipmap.c",
                    "line": 200,
                    "text": "if (l < ZIPMAP_BIGLEN && s != 1) return 0;  /* no l==0 check */",
                },
                {
                    "file": "src/zipmap.c",
                    "line": 240,
                    "text": "l = zipmapDecodeLength(p); p += l + ZIPMAP_LEN_BYTES(l) + ...;",
                },
                {
                    "file": "src/rdb.c",
                    "line": 2408,
                    "text": 'serverLog(LL_WARNING,"Hash zipmap with dup elements, or big length (%d)",zm_len);',
                },
            ]
        },
    ),
    _mock(
        "search_bash",
        "Read-only rg/bash search over source and tests.",
        {
            "output": (
                "src/zipmap.c:196:s = zipmapGetEncodedLengthSize(p);\n"
                "src/zipmap.c:197:l = zipmapDecodeLength(p);\n"
                "src/zipmap.c:240:return zipmapEncodeLength(NULL, l) + l;\n"
                "tests/unit/dump.tcl:123:set payload {fe 03 00 00 00}\n"
            )
        },
    ),
    _mock(
        "get_code_file_structure",
        "Return the directory/file tree for a project path.",
        {"files": ["src/zipmap.c", "src/zipmap.h", "src/rdb.c", "src/rdb.h"]},
    ),
    _mock(
        "fetch_file",
        "Fetch raw file content by path.",
        {
            "content": (
                "/* zipmapValidateIntegrity: validates a raw zipmap blob. */\n"
                "int zipmapValidateIntegrity(unsigned char *p, size_t len, int deep) {\n"
                "    unsigned int l;\n"
                "    unsigned char *e = p + len;\n"
                "    while (p < e) {\n"
                "        l = zipmapDecodeLength(p);  /* returns 0 for crafted input */\n"
                "        /* BUG: no check for l == 0 */\n"
                "        if (l < ZIPMAP_BIGLEN && p[0] != l) return 0;\n"
                "        p += zipmapRawKeyLength(p);\n"
                "        /* value */\n"
                "        l = zipmapDecodeLength(p);\n"
                "        /* BUG: no check for l == 0 here either */\n"
                "        p += zipmapRawValueLength(p);\n"
                "    }\n"
                "    return 1;\n"
                "}\n"
            )
        },
    ),
    # DAP tools — all return no_tunnel so Phase 5 is skipped (expected in smoke)
    *[
        _mock(name, f"DAP: {name}", {"error_type": "no_tunnel", "available": False})
        for name in [
            "set_breakpoints",
            "start_debug_session",
            "take_debug_snapshot",
            "step_over",
            "step_into",
            "step_out",
            "continue_execution",
            "evaluate_expression",
            "list_debug_sessions",
            "stop_debug_session",
        ]
    ],
    _mock(
        "run_validation",
        "Run a validation command and return pass/fail with evidence.",
        {
            "status": "error",
            "exit_code": 1,
            "evidence_summary": "Command not available in smoke-test mode.",
        },
    ),
]


# ---------------------------------------------------------------------------
# LLM model builder
# ---------------------------------------------------------------------------
def _build_llm(model_name: str) -> Any:
    """Return a pydantic_ai model from whichever provider API key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from pydantic_ai.models.anthropic import AnthropicModel
        return AnthropicModel(model_name)
    if os.environ.get("OPENAI_API_KEY"):
        from pydantic_ai.models.openai import OpenAIModel
        return OpenAIModel(model_name or "gpt-4o-mini")
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        from pydantic_ai.models.gemini import GeminiModel
        return GeminiModel(model_name or "gemini-2.0-flash")
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY)."
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
async def run_smoke(model_name: str) -> tuple[str, str, list[str], list[dict], float]:
    """Run the debug agent against DEBUGGING_SIGNAL.

    Returns
    -------
    (final_output, full_text, tool_call_names, hypotheses, elapsed_seconds)

    full_text   — ALL text across every LLM turn, joined with newlines.  The
                  **Debugger:** status line is emitted in an intermediate turn
                  and would be missed if we checked result.output alone.
    hypotheses  — Snapshot taken INSIDE the async context (ContextVar goes out
                  of scope after asyncio.run() returns).
    """
    store = HypothesisStore(conversation_id="smoke-test")
    _hypothesis_store_ctx.set(store)

    real_hyp_tools = wrap_structured_tools(create_hypothesis_state_tools())
    all_tools = real_hyp_tools + _MOCK_TOOLS

    agent = Agent(
        model=_build_llm(model_name),
        tools=all_tools,
        system_prompt=debug_task_prompt,
    )

    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        agent.run(DEBUGGING_SIGNAL),
        timeout=float(os.environ.get("SMOKE_TIMEOUT", "120")),
    )
    elapsed = time.perf_counter() - t0

    # Collect ALL text and tool calls from the full conversation history
    tool_calls: list[str] = []
    text_parts: list[str] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_calls.append(part.tool_name)
                elif isinstance(part, TextPart) and part.content:
                    text_parts.append(part.content)

    full_text = "\n".join(text_parts)

    # Snapshot the store BEFORE leaving this async context
    hypotheses = [r.model_dump(mode="json") for r in store.list_all()]

    return result.output, full_text, tool_calls, hypotheses, elapsed


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def print_report(
    model_name: str,
    final_output: str,
    full_text: str,
    tool_calls: list[str],
    hypotheses: list[dict],
    checks: dict[str, bool],
    elapsed: float,
) -> bool:
    """Print a human-readable smoke-test report. Returns True if all checks pass."""
    tty = sys.stdout.isatty()
    G = "\033[92m" if tty else ""
    R = "\033[91m" if tty else ""
    Y = "\033[93m" if tty else ""
    B = "\033[1m" if tty else ""
    E = "\033[0m" if tty else ""

    all_passed = all(checks.values())
    status = f"{G}PASSED{E}" if all_passed else f"{R}FAILED{E}"

    print(f"\n{B}{'='*60}{E}")
    print(f"{B}  Debug Agent Smoke Test{E}")
    print(f"{B}{'='*60}{E}")
    print(f"  Model   : {model_name}")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Status  : {status}")
    print()

    print(f"{B}  Tool Call Sequence ({len(tool_calls)} calls):{E}")
    for i, name in enumerate(tool_calls, 1):
        print(f"    {i:2}. {name}")
    print()

    print(f"{B}  Protocol Compliance:{E}")
    for name, passed in checks.items():
        icon = f"{G}✓{E}" if passed else f"{R}✗{E}"
        label = name.replace("_", " ")
        print(f"    {icon}  {label}")
    print()

    print(f"{B}  Recorded Hypotheses ({len(hypotheses)}):{E}")
    if hypotheses:
        for h in hypotheses:
            evidence_count = len(h.get("evidence", []))
            print(
                f"    [{h['id']}] {h['title'][:65]}"
                f"\n           status={h['status']}, evidence_items={evidence_count}"
            )
    else:
        print(f"    {R}(none — record_hypothesis was not called or failed){E}")
    print()

    print(f"{B}  Full Conversation Text (first 1000 chars):{E}")
    preview = full_text[:1000]
    print(f"{Y}{preview}{E}")
    if len(full_text) > 1000:
        print(f"  ... [{len(full_text) - 1000} more chars]")
    print()

    failed_checks = [k for k, v in checks.items() if not v]
    if failed_checks:
        print(f"{R}{B}  Failed checks:{E}")
        for name in failed_checks:
            print(f"  {R}  ✗  {name.replace('_', ' ')}{E}")
        print()

    print(f"{B}{'='*60}{E}\n")
    return all_passed


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------
_has_api_key = bool(
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)


@pytest.mark.smoke
@pytest.mark.skipif(not _has_api_key, reason="No LLM API key — set ANTHROPIC_API_KEY")
def test_debug_agent_follows_hypothesis_protocol() -> None:
    """Smoke: real LLM must follow the hypothesis-driven debugging protocol.

    Feeds the ASan heap-buffer-overflow signal from valkey-io/valkey and
    asserts all OUTPUT CONTRACT requirements from debug_task_prompt are met.

    Skipped automatically when no LLM API key is in the environment.
    """
    model_name = os.environ.get("SMOKE_MODEL", "claude-haiku-4-5-20251001")
    final_output, full_text, tool_calls, hypotheses, elapsed = asyncio.run(run_smoke(model_name))
    checks = check_protocol_compliance(full_text, tool_calls, hypotheses)
    print_report(model_name, final_output, full_text, tool_calls, hypotheses, checks, elapsed)

    failed = {k for k, v in checks.items() if not v}
    assert not failed, (
        f"Agent failed {len(failed)} protocol check(s):\n"
        + "\n".join(f"  ✗ {c.replace('_', ' ')}" for c in sorted(failed))
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug agent smoke test")
    parser.add_argument(
        "--model",
        default=os.environ.get("SMOKE_MODEL", "claude-haiku-4-5-20251001"),
        help="LLM model name override",
    )
    args = parser.parse_args()

    if not _has_api_key:
        print(
            "ERROR: No LLM API key found.\n"
            "Set ANTHROPIC_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    final_output, full_text, tool_calls, hypotheses, elapsed = asyncio.run(run_smoke(args.model))
    checks = check_protocol_compliance(full_text, tool_calls, hypotheses)
    passed = print_report(args.model, final_output, full_text, tool_calls, hypotheses, checks, elapsed)
    sys.exit(0 if passed else 1)
