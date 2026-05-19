"""Unit tests for the debug hypothesis markdown contract module.

Verifies the contract's enum order, section headers, discovery priority,
and the worked example — so any accidental change to the single source of
truth breaks loudly before it propagates to the VS Code webview parser.
"""
import pytest

from app.modules.intelligence.agents.chat_agents.system_agents.debug_hypothesis_contract import (
    HypothesisStatus,
    HYPOTHESIS_STATUS_ENUM,
    HYPOTHESIS_SECTION_HEADERS,
    DISCOVERY_PRIORITY_ORDER,
    HYPOTHESIS_MARKDOWN_EXAMPLE,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. HypothesisStatus enum
# ---------------------------------------------------------------------------

EXPECTED_STATUS_ORDER = [
    "proposed",
    "debugging",
    "needs_evidence",
    "supported",
    "rejected",
    "fix_proposed",
    "needs_revision",
    "validated",
]


def test_hypothesis_status_has_exactly_eight_members():
    assert len(HypothesisStatus) == 8


def test_hypothesis_status_members_are_in_documented_order():
    actual = [member.value for member in HypothesisStatus]
    assert actual == EXPECTED_STATUS_ORDER


def test_hypothesis_status_values_are_accessible_by_name():
    for name in EXPECTED_STATUS_ORDER:
        member = HypothesisStatus(name)
        assert member.value == name


def test_hypothesis_status_enum_alias_is_same_class():
    assert HYPOTHESIS_STATUS_ENUM is HypothesisStatus


# ---------------------------------------------------------------------------
# 2. HYPOTHESIS_SECTION_HEADERS
# ---------------------------------------------------------------------------

def test_section_headers_contains_title_format_string():
    title_fmt = HYPOTHESIS_SECTION_HEADERS["title"]
    # Must contain placeholders for n and title
    assert "{n}" in title_fmt
    assert "{title}" in title_fmt
    assert title_fmt.startswith("## Hypothesis")


def test_section_headers_contains_status_format_string():
    status_fmt = HYPOTHESIS_SECTION_HEADERS["status"]
    assert "{status}" in status_fmt
    assert status_fmt.startswith("### Status:")


def test_section_headers_contains_evidence_literal():
    assert HYPOTHESIS_SECTION_HEADERS["evidence"] == "### Evidence"


def test_section_headers_contains_validation_plan_literal():
    assert HYPOTHESIS_SECTION_HEADERS["validation_plan"] == "### Validation Plan"


def test_section_headers_contains_debugger_evidence_literal():
    assert HYPOTHESIS_SECTION_HEADERS["debugger_evidence"] == "### Debugger Evidence"


def test_section_headers_contains_fix_proposal_literal():
    assert HYPOTHESIS_SECTION_HEADERS["fix_proposal"] == "### Fix Proposal"


# ---------------------------------------------------------------------------
# 3. DISCOVERY_PRIORITY_ORDER
# ---------------------------------------------------------------------------

def test_discovery_priority_order_starts_with_context_graph():
    assert DISCOVERY_PRIORITY_ORDER[0] == "query_context_graph"


def test_discovery_priority_order_contains_all_expected_tools():
    expected = [
        "query_context_graph",
        "ask_knowledge_graph_queries",
        "search_text",
        "get_code_file_structure",
        "fetch_file",
    ]
    assert DISCOVERY_PRIORITY_ORDER == expected


# ---------------------------------------------------------------------------
# 4. HYPOTHESIS_MARKDOWN_EXAMPLE
# ---------------------------------------------------------------------------

def test_example_contains_hypothesis_title_pattern():
    assert "## Hypothesis" in HYPOTHESIS_MARKDOWN_EXAMPLE
    # The worked example title
    assert "Payment timeout" in HYPOTHESIS_MARKDOWN_EXAMPLE


def test_example_status_line_value_is_a_valid_enum_member():
    lines = HYPOTHESIS_MARKDOWN_EXAMPLE.splitlines()
    status_line = next(
        (line for line in lines if line.startswith("### Status:")),
        None,
    )
    assert status_line is not None, "No '### Status:' line found in example"
    status_value = status_line.split(":", 1)[1].strip()
    # Must be parseable as a HypothesisStatus member
    member = HypothesisStatus(status_value)
    assert member.value == status_value


def test_example_contains_evidence_section():
    assert "### Evidence" in HYPOTHESIS_MARKDOWN_EXAMPLE


def test_example_contains_validation_plan_section():
    assert "### Validation Plan" in HYPOTHESIS_MARKDOWN_EXAMPLE
