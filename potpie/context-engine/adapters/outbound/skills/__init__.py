"""Skill catalog + per-harness install-target adapters."""

from __future__ import annotations

from adapters.outbound.skills.bundle_catalog import (
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
    "ClaudeAgentTarget",
    "CodexAgentTarget",
    "CursorAgentTarget",
    "OpenCodeAgentTarget",
    "ProjectAgentTarget",
    "catalog_by_id",
    "load_bundle_skills",
    "recommended_skill_ids",
]
