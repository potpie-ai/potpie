"""VERIFY subagent (diff and validation)."""

from __future__ import annotations

from pydantic_deep.types import SubAgentConfig

from poc.config.provider import get_model_settings
from poc.config.settings import MODEL_MAX_CONCURRENCY
from poc.tools.toolsets_builder import verification_toolset

ROLE = "Verification and Rollup Specialist"


def subagent_config() -> SubAgentConfig:
    return {
        "name": "verify",
        "description": (
            "Runs targeted validation, reviews the staged diff, and reports coverage gaps."
        ),
        "instructions": (
            f"You are the {ROLE}. "
            "BUDGET: You have at most 10 tool calls. "
            "Do not make exploratory changes. "
            "Inspect the staged diff with show_diff and get_changes_summary. "
            "Run targeted validation with validate_only_bash (pytest specific files, py_compile, compileall). "
            "Verify: files compile, tests pass, no stale references, diff matches objective. "
            "You MUST call record_verification_result with PASS or FAIL before your final response. "
            "Report: what passed, what failed, remaining risks. "
            "Only suggest additional edits if validation proves a concrete defect. "
            "End with a ## Task Result section with PASS or FAIL status."
        ),
        "toolsets": [verification_toolset()],
        "can_ask_questions": True,
        "max_questions": 1,
        "typical_complexity": "moderate",
        "typically_needs_context": True,
        "agent_kwargs": {
            "model_settings": get_model_settings(),
            "max_concurrency": MODEL_MAX_CONCURRENCY,
        },
    }
