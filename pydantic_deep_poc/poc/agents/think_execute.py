"""THINK_EXECUTE subagent (execute tool suite)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.tools.toolsets_builder import think_execute_toolset

MAX_ITER = 20

ROLE = "Code Implementation and Review Specialist"
GOAL = "Implement code solutions and review them for quality"
BACKSTORY = (
    "You are a skilled developer who excels at writing clean, efficient, maintainable code "
    "and reviewing it for quality and best practices."
)
TASK = (
    "Implement code solutions following best practices and project patterns, "
    "then review for quality, security, and maintainability"
)
EXPECTED_OUTPUT = (
    "Production-ready code implementation with proper error handling and quality review"
)


def subagent_config() -> SubAgentConfig:
    return {
        "name": "think_execute",
        "description": "Implements and reviews code in the repo using full execute tool suite.",
        "instructions": (
            f"You are {ROLE}. {TASK} "
            "End with a ## Task Result summary."
        ),
        "toolsets": [think_execute_toolset()],
        "extra": {"max_iter": MAX_ITER},
    }
