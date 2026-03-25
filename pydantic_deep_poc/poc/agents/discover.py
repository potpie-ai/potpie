"""DISCOVER subagent (bounded read-only discovery)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.config.provider import get_model_settings
from poc.config.settings import MODEL_MAX_CONCURRENCY
from poc.tools.toolsets_builder import discovery_toolset

ROLE = "Codebase Discovery Specialist"


def subagent_config() -> SubAgentConfig:
    return {
        "name": "discover",
        "description": (
            "Performs read-only bounded discovery, produces impacted files, known findings, "
            "and implementation slices."
        ),
        "instructions": (
            f"You are the {ROLE}. "
            "You are read-only. Do not propose or write final code changes. "
            "BUDGET: You have at most 8 tool calls and 2 turns. Do not exceed this. "
            "Use read_only_bash for search and inspection (rg, grep, find, ls, cat, git status/diff/log). "
            "Use file_ops for reading code files. "
            "Use web search for external library documentation. "
            "Discover the minimum needed context and return a compact brief with: "
            "impacted files, findings, risks, and suggested slices. "
            "Do not re-explore areas already described in the task packet. "
            "If critical context is missing, ask_parent once with one specific question. "
            "If ask_parent indicates there is no live parent channel, stop and return BLOCKED. "
            "End with a ## Task Result section."
        ),
        "toolsets": [discovery_toolset()],
        "can_ask_questions": True,
        "max_questions": 1,
        "typical_complexity": "moderate",
        "typically_needs_context": True,
        "agent_kwargs": {
            "model_settings": get_model_settings(),
            "max_concurrency": MODEL_MAX_CONCURRENCY,
        },
    }
