"""Markdown contract for transporting debugging hypotheses from the agent to the VS Code webview.

This contract is consumed by both the agent system prompt and the VS Code webview parser.
Any change here is a coordinated change across both sides.
"""
from enum import Enum

__all__ = [
    "HypothesisStatus",
    "HYPOTHESIS_STATUS_ENUM",
    "HYPOTHESIS_SECTION_HEADERS",
    "DISCOVERY_PRIORITY_ORDER",
    "HYPOTHESIS_MARKDOWN_EXAMPLE",
]


class HypothesisStatus(str, Enum):
    """Lifecycle states a debugging hypothesis can occupy, in order of progression."""

    PROPOSED = "proposed"
    DEBUGGING = "debugging"
    NEEDS_EVIDENCE = "needs_evidence"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    FIX_PROPOSED = "fix_proposed"
    NEEDS_REVISION = "needs_revision"
    VALIDATED = "validated"


# Spec-named alias for cross-side parsers (VS Code webview / external consumers) that reference the canonical name from the contract spec.
HYPOTHESIS_STATUS_ENUM = HypothesisStatus


# ---------------------------------------------------------------------------
# Section header strings — the agent must emit these verbatim.
# Format strings use Python str.format() placeholders.
# ---------------------------------------------------------------------------

HYPOTHESIS_SECTION_HEADERS = {
    # e.g. "## Hypothesis 1: Payment timeout is not converted into a controlled response"
    "title": "## Hypothesis {n}: {title}",
    # e.g. "### Status: debugging"
    "status": "### Status: {status}",
    # Literal headers — always present in a complete hypothesis block.
    "evidence": "### Evidence",
    "validation_plan": "### Validation Plan",
    # Populated during the active debugging phase.
    "debugger_evidence": "### Debugger Evidence",
    # Only emitted when status is fix_proposed or later.
    "fix_proposal": "### Fix Proposal",
}


# ---------------------------------------------------------------------------
# Code-discovery priority order.
# The agent must consult tools in this sequence when locating code.
# query_context_graph (A8) is listed first so the agent is forward-compatible
# with the context graph once it becomes available; it may return
# available=False today and the agent falls through to the next tool.
# ---------------------------------------------------------------------------

DISCOVERY_PRIORITY_ORDER: list[str] = [
    "query_context_graph",         # A8 - primary, may return available=False today
    "ask_knowledge_graph_queries", # existing, requires non-INFERRING status
    # search_text is the registered grep-style text search tool; swap for search_colgrep when feat/colgrep merges
    "search_text",                 # local_mode_only: ripgrep-backed text search via VS Code tunnel
    "get_code_file_structure",     # directory/file tree for structural navigation
    "fetch_file",                  # fetch raw file content by path
]


# ---------------------------------------------------------------------------
# Canonical worked example — one fully-formed hypothesis block.
# This is the reference for prompt writers and webview parser implementers.
# ---------------------------------------------------------------------------

# Generic structural skeleton — STRUCTURE ONLY.
# Every concrete word is a placeholder in angle brackets so the model cannot accidentally
# borrow domain vocabulary from this example. The trailing `---` is the card terminator the
# VS Code parser keys on; every emitted hypothesis card must end with it.
HYPOTHESIS_MARKDOWN_EXAMPLE = """\
## Hypothesis 1: <one-line statement of the suspected root cause>

### Status: proposed

### Evidence

- <observation tied to a specific file:line, symbol, or quoted fragment of the failure signal>
- <observation that connects the observed failure to the suspected origin>
- <observation showing the gap between expected and actual behavior at that location>

### Validation Plan

- <breakpoint or instrumentation step naming a concrete file:line and what to inspect>
- <command, test, or repro step that exercises the suspected path>
- <what observation at the breakpoint will confirm or refute this hypothesis>

---
"""
