"""GitHub integration subagent."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.tools.toolsets_builder import github_toolset

MAX_ITER = 15

ROLE = "GitHub Integration Specialist"
GOAL = "Handle all GitHub repository operations including PRs, branches, and commits"
BACKSTORY = (
    "You are a specialized agent for GitHub operations. You handle pull requests, branches, "
    "file updates, and PR comments efficiently in an isolated context."
)


def subagent_config() -> SubAgentConfig:
    return {
        "name": "github",
        "description": "GitHub / PR operations (partially real git, stub API).",
        "instructions": (
            "Execute GitHub operations as requested by the supervisor. Use GitHub tools to create "
            "branches, PRs, update files, and add comments. "
            'Return results in "## Task Result" format with PR numbers, branch names, and GitHub URLs. '
            "Prefer create_pr_workflow when the user confirms a PR."
        ),
        "toolsets": [github_toolset()],
        "extra": {"max_iter": MAX_ITER},
    }
