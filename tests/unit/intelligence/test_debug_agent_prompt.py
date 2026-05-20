"""Smoke tests for the rewritten debug_task_prompt (A1.4).

These tests verify that the prompt string wires in all the tools and contract
constants from the A2/A3/A4/A5/A6/A8 work. They are NOT "did the agent reason
well" tests — they are "did the prompt actually mention the tools" checks.
"""
import pytest

from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent_prompt import (
    debug_task_prompt,
)
from app.modules.intelligence.agents.chat_agents.system_agents.debug_hypothesis_contract import (
    HYPOTHESIS_MARKDOWN_EXAMPLE,
    HypothesisStatus,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# 1. All tool names are mentioned
# ---------------------------------------------------------------------------

REQUIRED_TOOLS = [
    # A2 — failure signal parser
    "parse_failure_signal",
    # A4 — workspace debug context
    "get_workspace_debug_context",
    # A8 — context graph (discovery priority 1)
    "query_context_graph",
    # knowledge graph (discovery priority 2)
    "ask_knowledge_graph_queries",
    # text search (discovery priority 3)
    "search_text",
    # structural navigation (discovery priority 4)
    "get_code_file_structure",
    # raw file fetch (discovery priority 5)
    "fetch_file",
    # A5 — hypothesis state tools
    "record_hypothesis",
    "update_hypothesis_status",
    "append_hypothesis_evidence",
    "list_hypotheses",
    # A3 — DAP debugger tools
    "set_breakpoints",
    "start_debug_session",
    "take_debug_snapshot",
    "step_over",
    "step_into",
    "step_out",
    "continue_execution",
    "evaluate_expression",
    # A6 — validation
    "run_validation",
]


@pytest.mark.parametrize("tool_name", REQUIRED_TOOLS)
def test_prompt_mentions_tool(tool_name: str) -> None:
    assert tool_name in debug_task_prompt, (
        f"Expected tool '{tool_name}' to appear in debug_task_prompt but it was not found."
    )


# ---------------------------------------------------------------------------
# 2. Markdown contract example is embedded
# ---------------------------------------------------------------------------

# Sentinel string from HYPOTHESIS_MARKDOWN_EXAMPLE that is unique to the skeleton.
# The placeholder text is intentionally generic; the angle-bracket phrase below is
# specific to the canonical skeleton.
_EXAMPLE_SENTINEL = "<one-line statement of the suspected root cause>"


def test_prompt_embeds_hypothesis_markdown_example() -> None:
    assert _EXAMPLE_SENTINEL in debug_task_prompt, (
        "Expected a substring of HYPOTHESIS_MARKDOWN_EXAMPLE "
        f"({_EXAMPLE_SENTINEL!r}) to appear in debug_task_prompt."
    )


# ---------------------------------------------------------------------------
# 3. Every status enum value is mentioned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", list(HypothesisStatus))
def test_prompt_mentions_status_value(status: HypothesisStatus) -> None:
    assert status.value in debug_task_prompt, (
        f"Expected hypothesis status '{status.value}' to appear in debug_task_prompt."
    )


# ---------------------------------------------------------------------------
# 4. Tunnel-unavailable handling is present
# ---------------------------------------------------------------------------

def test_prompt_mentions_no_tunnel_handling() -> None:
    # At least one of these phrases must be present.
    phrases = ["no_tunnel", "extension is not connected", "VS Code extension is not connected"]
    found = any(phrase in debug_task_prompt for phrase in phrases)
    assert found, (
        "Expected the prompt to mention tunnel-unavailable handling "
        f"(one of: {phrases}). None were found."
    )


# ---------------------------------------------------------------------------
# 5. OUTPUT CONTRACT section — top-of-prompt guardrails
# ---------------------------------------------------------------------------

def test_prompt_has_output_contract_section() -> None:
    """The OUTPUT CONTRACT must appear near the top of the prompt so the model reads
    it before the per-phase instructions."""
    assert "OUTPUT CONTRACT" in debug_task_prompt
    # And it must appear before the first phase header so the model encounters it first.
    contract_idx = debug_task_prompt.find("OUTPUT CONTRACT")
    phase1_idx = debug_task_prompt.find("Phase 1")
    assert contract_idx < phase1_idx, (
        "OUTPUT CONTRACT must appear before Phase 1 in the prompt."
    )


def test_prompt_has_forbidden_output_list() -> None:
    """The prompt must explicitly forbid the conclusion-section shapes that previous
    runs degraded into (CVE writeup / Analysis Summary / etc.)."""
    assert "FORBIDDEN OUTPUT" in debug_task_prompt
    forbidden_section_titles = [
        "Analysis Summary",
        "Final Summary",
        "Root Cause Analysis",
        "Vulnerability Overview",
        "Recommended Fix",
    ]
    for title in forbidden_section_titles:
        assert title in debug_task_prompt, (
            f"Expected the FORBIDDEN OUTPUT list to mention {title!r}."
        )


def test_prompt_requires_debugger_state_emission() -> None:
    """Phase 2 must require the model to emit explicit debugger-state acknowledgement
    so the user can see when Phase 5 will be skipped."""
    available_marker = "**Debugger:** available"
    unavailable_marker = "**Debugger:** unavailable"
    assert available_marker in debug_task_prompt, (
        f"Expected required available marker {available_marker!r} in the prompt."
    )
    assert unavailable_marker in debug_task_prompt, (
        f"Expected required unavailable marker {unavailable_marker!r} in the prompt."
    )


def test_prompt_has_refinement_rule() -> None:
    """If the agent refines its theory after Phase 4, it must call record_hypothesis
    again rather than write a conclusion based on an un-recorded theory."""
    assert "REFINEMENT RULE" in debug_task_prompt, (
        "Expected the prompt to include a REFINEMENT RULE clause."
    )


def test_prompt_makes_phase_1_required() -> None:
    """Phase 1 (parse_failure_signal) must be flagged as required, since past runs
    silently skipped it.

    Anchors on '### Phase 1' (the actual heading) rather than 'Phase 1', since the
    OUTPUT CONTRACT section earlier in the prompt back-references later phases by
    name.
    """
    phase1_idx = debug_task_prompt.find("### Phase 1")
    phase2_idx = debug_task_prompt.find("### Phase 2")
    assert phase1_idx != -1, "Expected a '### Phase 1' heading in the prompt."
    assert phase2_idx > phase1_idx, "Expected '### Phase 2' to follow '### Phase 1'."
    phase1_section = debug_task_prompt[phase1_idx:phase2_idx]
    assert "REQUIRED" in phase1_section, (
        "Phase 1 section must be flagged REQUIRED so the model cannot skip it."
    )


def test_prompt_requires_card_terminator() -> None:
    """The Phase 4 description must instruct the model to terminate each hypothesis
    card with a literal '---' line so the webview parser can identify boundaries.

    Anchors on the heading prefix to avoid OUTPUT CONTRACT back-references.
    """
    phase4_idx = debug_task_prompt.find("### Phase 4")
    phase5_idx = debug_task_prompt.find("### Phase 5")
    assert phase4_idx != -1, "Expected a '### Phase 4' heading in the prompt."
    assert phase5_idx > phase4_idx, "Expected '### Phase 5' to follow '### Phase 4'."
    phase4_section = debug_task_prompt[phase4_idx:phase5_idx]
    assert "---" in phase4_section
    assert "terminator" in phase4_section.lower(), (
        "Phase 4 must explicitly call the '---' line a card terminator."
    )
