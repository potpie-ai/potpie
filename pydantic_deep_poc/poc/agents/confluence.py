"""Confluence integration subagent (stub tools)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.tools.toolsets_builder import confluence_toolset

MAX_ITER = 15

ROLE = "Confluence Integration Specialist"
GOAL = "Handle all Confluence documentation operations including pages, spaces, and search"
BACKSTORY = (
    "You are a specialized agent for Confluence operations. You handle page creation, updates, "
    "searches, and comments efficiently in an isolated context."
)


def subagent_config() -> SubAgentConfig:
    return {
        "name": "confluence",
        "description": "Confluence documentation (stubbed).",
        "instructions": (
            "Execute Confluence operations as requested by the supervisor. Use Confluence tools to "
            "search spaces, get/create/update pages, and add comments. "
            'Return results in "## Task Result" format with page IDs, titles, and Confluence URLs.'
        ),
        "toolsets": [confluence_toolset()],
        "extra": {"max_iter": MAX_ITER},
    }
