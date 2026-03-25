"""Linear integration subagent (stub tools)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.tools.toolsets_builder import linear_toolset

MAX_ITER = 15

ROLE = "Linear Integration Specialist"
GOAL = "Handle Linear issue management operations including fetching and updating issues"
BACKSTORY = (
    "You are a specialized agent for Linear operations. You handle issue fetching and updates "
    "efficiently in an isolated context."
)


def subagent_config() -> SubAgentConfig:
    return {
        "name": "linear",
        "description": "Linear issues (stubbed).",
        "instructions": (
            "Execute Linear operations as requested by the supervisor. Use Linear tools to get issue "
            "details and update issue fields. "
            'Return results in "## Task Result" format with issue IDs, titles, and Linear URLs.'
        ),
        "toolsets": [linear_toolset()],
        "extra": {"max_iter": MAX_ITER},
    }
