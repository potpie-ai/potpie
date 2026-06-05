"""Skill catalog + per-harness install-target adapters."""

from __future__ import annotations

from adapters.outbound.skills.builtin_catalog import (
    BUILTIN_SKILLS,
    RECOMMENDED_SKILL_IDS,
    catalog_by_id,
)
from adapters.outbound.skills.claude_target import ClaudeAgentTarget

__all__ = [
    "BUILTIN_SKILLS",
    "RECOMMENDED_SKILL_IDS",
    "ClaudeAgentTarget",
    "catalog_by_id",
]
