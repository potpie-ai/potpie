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

# The worked example below shows the mandatory sections only (Evidence + Validation Plan).
# Optional sections (Debugger Evidence, Fix Proposal) are referenced in
# HYPOTHESIS_SECTION_HEADERS and are populated by the agent during the debugging and
# fix_proposed phases respectively.
HYPOTHESIS_MARKDOWN_EXAMPLE = """\
## Hypothesis 1: Payment timeout is thrown but not converted into a controlled checkout response

### Status: debugging

### Evidence

- Stack trace captured from Sentry includes `paymentAdapter.chargeCard` at the top of the call
  chain, indicating the timeout originates inside the payment adapter layer.
- `createOrder` surfaces the exception directly as an HTTP 500 rather than mapping it to a
  controlled checkout failure response (e.g. `PaymentDeclinedError`).
- The related integration test `should return payment_failed on timeout` asserts a controlled
  `payment_failed` response body — this test is currently red, confirming the gap.

### Validation Plan

- Set a breakpoint in `chargeCard` at the point where the timeout exception is raised or caught
  to confirm whether any local handling occurs before the exception propagates.
- Set a breakpoint in `createOrder`'s error-handling block to verify whether a timeout error
  is caught and mapped, or escapes uncaught.
- Run the checkout reproduction script with a mocked slow payment gateway to trigger the timeout
  path deterministically.
- Inspect the exception object at each breakpoint: confirm it is a raw timeout error rather than
  a domain-typed `PaymentTimeoutError`, then trace whether any conversion to a controlled
  response ever takes place.
"""
