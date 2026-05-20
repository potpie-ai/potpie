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


# ---------------------------------------------------------------------------
# 6. Multi-turn flow — pauses + missing-command UX
# ---------------------------------------------------------------------------

def test_prompt_has_multi_turn_flow_section() -> None:
    """The OUTPUT CONTRACT must explain that the loop is multi-turn with explicit pauses."""
    assert "MULTI-TURN FLOW" in debug_task_prompt, (
        "Expected a MULTI-TURN FLOW section in the OUTPUT CONTRACT."
    )


def test_phase_4_pauses_before_dap_tools() -> None:
    """Phase 4 must end with a pause for user input before any DAP tool runs."""
    phase4_idx = debug_task_prompt.find("### Phase 4")
    phase5_idx = debug_task_prompt.find("### Phase 5")
    phase4_section = debug_task_prompt[phase4_idx:phase5_idx]
    # Must instruct the model to NOT call DAP tools in the Phase 4 turn.
    assert "STOP" in phase4_section, (
        "Phase 4 must contain a STOP instruction at the end."
    )
    assert "set_breakpoints" in phase4_section and "Do NOT call" in phase4_section, (
        "Phase 4 must explicitly forbid DAP tool calls in the same turn."
    )


def test_phase_4_allows_one_to_four_hypotheses() -> None:
    """The mandatory count is now 1-4 (not 2-4); a single confident hypothesis is fine."""
    phase4_idx = debug_task_prompt.find("### Phase 4")
    phase5_idx = debug_task_prompt.find("### Phase 5")
    phase4_section = debug_task_prompt[phase4_idx:phase5_idx]
    assert "1 to 4" in phase4_section, (
        "Phase 4 count should be '1 to 4 hypotheses' — single hypothesis allowed."
    )


def test_phase_5_entry_requires_list_hypotheses() -> None:
    """Phase 5 runs in a new turn after the user picks; first action must be
    list_hypotheses() to recover persisted state."""
    phase5_idx = debug_task_prompt.find("### Phase 5")
    phase6_idx = debug_task_prompt.find("### Phase 6")
    assert phase5_idx != -1 and phase6_idx > phase5_idx
    phase5_section = debug_task_prompt[phase5_idx:phase6_idx]
    assert "list_hypotheses()" in phase5_section, (
        "Phase 5 must require a list_hypotheses() call at entry to recover state."
    )


def test_phase_5c_has_three_cases() -> None:
    """Phase 5c must enumerate three sub-cases: config available, missing command, no tunnel.

    The 'missing command' case is the new one requested: when the workspace has no
    launch.json and no inferred commands, agent asks user with two clearly framed options.
    """
    phase5_idx = debug_task_prompt.find("### Phase 5")
    phase6_idx = debug_task_prompt.find("### Phase 6")
    phase5_section = debug_task_prompt[phase5_idx:phase6_idx]
    # Three case markers
    assert "Case A" in phase5_section
    assert "Case B" in phase5_section
    assert "Case C" in phase5_section
    # Missing-command path mentions the two options the user described
    assert "Tell me a command to run" in phase5_section, (
        "Case B must offer the 'user supplies command' option."
    )
    assert "propose a debug command" in phase5_section, (
        "Case B must offer the 'agent proposes a command' option."
    )


def test_rejected_verdict_pauses_for_user() -> None:
    """After a 'rejected' verdict, agent must pause and ask user, not auto-move."""
    phase5_idx = debug_task_prompt.find("### Phase 5")
    phase6_idx = debug_task_prompt.find("### Phase 6")
    phase5_section = debug_task_prompt[phase5_idx:phase6_idx]
    # Find the 'rejected' sub-section
    rejected_idx = phase5_section.find('status="rejected"')
    assert rejected_idx != -1
    after_rejected = phase5_section[rejected_idx:]
    # Must contain a pause + question
    assert "STOP" in after_rejected, (
        "Rejected verdict must STOP the response, not auto-move to next hypothesis."
    )
    assert ("revise hypotheses" in after_rejected.lower()
            or "try h" in after_rejected.lower()), (
        "Rejected verdict must offer the user options to revise or move on."
    )


def test_prompt_has_all_of_them_escape_hatch() -> None:
    """User can override pauses by saying 'all of them' at Phase 4 exit."""
    assert "all of them" in debug_task_prompt, (
        "Prompt must support an 'all of them' escape hatch so users can run "
        "through hypotheses without pausing if they prefer."
    )
