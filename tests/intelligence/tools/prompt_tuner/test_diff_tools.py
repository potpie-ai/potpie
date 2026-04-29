"""
Tests for the propose_prompt_diff and apply_prompt_change tools.
"""

from app.modules.intelligence.tools.prompt_tuner.diff_tools import (
    ProposePromptDiffTool,
)


def test_propose_diff_formats_correctly():
    tool = ProposePromptDiffTool(None, None)
    result = tool.run(
        root_cause="The prompt lacks explicit instruction to limit colgrep calls.",
        edits=[
            {
                "title": "Add search limit",
                "location": "After ## Tool Usage section",
                "old_text": "",
                "new_text": "Never call colgrep more than 2 times per query.",
                "rationale": "Prevents redundant semantic searches.",
            },
            {
                "title": "Clarify colgrep purpose",
                "location": "## Tool Usage > colgrep",
                "old_text": "Use colgrep to search the codebase.",
                "new_text": "Use colgrep for SEMANTIC search to identify candidate files (max 2 calls).",
                "rationale": "Makes the tool's purpose and limits explicit.",
            },
        ],
    )
    assert "Root Cause" in result
    assert "EDIT 1" in result
    assert "EDIT 2" in result
    assert "2 edits" in result
    assert "Add search limit" in result
    assert "Apply these changes?" in result
    assert "(new section)" in result
    assert "- Use colgrep to search the codebase." in result
    assert "+ Use colgrep for SEMANTIC search" in result


def test_propose_diff_single_edit():
    tool = ProposePromptDiffTool(None, None)
    result = tool.run(
        root_cause="Missing anti-pattern warning.",
        edits=[
            {
                "title": "Add anti-pattern",
                "location": "End of prompt",
                "old_text": "",
                "new_text": "Do NOT call colgrep with full sentences.",
                "rationale": "Agent sends verbose queries to semantic search.",
            },
        ],
    )
    assert "1 edits" in result
    assert "EDIT 1" in result
    assert "EDIT 2" not in result
