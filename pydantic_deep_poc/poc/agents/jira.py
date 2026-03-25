"""JIRA integration subagent (stub tools)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.tools.toolsets_builder import jira_toolset

MAX_ITER = 15

ROLE = "Jira Integration Specialist"
GOAL = "Handle all Jira operations including issue management, search, and workflow transitions"
BACKSTORY = (
    "You are a specialized agent for Jira operations. You handle issue creation, updates, "
    "searches, comments, and status transitions efficiently in an isolated context."
)


def subagent_config() -> SubAgentConfig:
    return {
        "name": "jira",
        "description": "Jira issue operations (stubbed in PoC).",
        "instructions": (
            "Execute Jira operations as requested by the supervisor. Use Jira tools to search, "
            "create, update, comment on, and transition issues. "
            'Return results in "## Task Result" format with issue keys and links.'
        ),
        "toolsets": [jira_toolset()],
        "extra": {"max_iter": MAX_ITER},
    }
