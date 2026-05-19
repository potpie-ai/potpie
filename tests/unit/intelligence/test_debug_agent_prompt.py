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

# Use a unique substring from HYPOTHESIS_MARKDOWN_EXAMPLE as the sentinel.
# "Payment timeout is thrown but not converted" appears only in the example.
_EXAMPLE_SENTINEL = "Payment timeout is thrown but not converted"


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
