"""Skill catalog + per-harness install-target adapters."""

from __future__ import annotations

from adapters.outbound.skills.bundle_catalog import (
    BUILTIN_SKILLS,
    RECOMMENDED_SKILL_IDS,
    catalog_by_id,
    load_bundle_skills,
    recommended_skill_ids,
)
from adapters.outbound.skills.claude_target import (
    ClaudeAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
    ProjectAgentTarget,
)

__all__ = [
    "BUILTIN_SKILLS",
    "RECOMMENDED_SKILL_IDS",
    "ClaudeAgentTarget",
    "CodexAgentTarget",
    "CursorAgentTarget",
    "OpenCodeAgentTarget",
    "ProjectAgentTarget",
    "catalog_by_id",
    "load_bundle_skills",
    "recommended_skill_ids",
]
