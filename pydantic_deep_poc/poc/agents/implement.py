"""IMPLEMENT subagent (CCM-only code changes)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.config.provider import get_model_settings
from poc.config.settings import MODEL_MAX_CONCURRENCY
from poc.tools.toolsets_builder import implementation_toolset

ROLE = "Code Implementation Specialist"


def subagent_config() -> SubAgentConfig:
    return {
        "name": "implement",
        "description": (
            "Implements a scoped code slice using CCM tools only, with narrow validation."
        ),
        "instructions": (
            f"You are the {ROLE}. "
            "BUDGET: You have at most 30 tool calls. Use them efficiently. "
            "SCOPE: Operate ONLY on the files listed in FILES_IN_SCOPE. "
            "Do not touch files outside your scope. "
            "Use Code Changes Manager tools for every code modification: "
            "- replace_in_file for targeted string replacements "
            "- update_file_lines for line-range edits "
            "- update_file_in_changes ONLY for small files (<10K chars) "
            "- add_file_to_changes for new files "
            "Use validate_only_bash for validation: pytest (specific files), py_compile, compileall. "
            "Do not use shell commands for file modifications (no sed -i, cp, mv, rm). "
            "Do not browse the web. Do not perform broad repo exploration. "
            "If a command fails, do not retry the same command. Try once differently or ask_parent. "
            "Do not clear, revert, or reset shared staged changes. "
            "If critical context is missing, ask_parent once with one precise question. "
            "If ask_parent indicates there is no live parent channel, stop and return BLOCKED. "
            "Return: files changed, validations run, and unresolved issues. "
            "End with a ## Task Result section."
        ),
        "toolsets": [implementation_toolset()],
        "can_ask_questions": True,
        "max_questions": 1,
        "typical_complexity": "complex",
        "typically_needs_context": True,
        "agent_kwargs": {
            "model_settings": get_model_settings(),
            "max_concurrency": MODEL_MAX_CONCURRENCY,
        },
    }
