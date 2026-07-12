"""Skill catalog + per-harness install-target adapters."""

from __future__ import annotations

from potpie_context_engine.adapters.outbound.skills.bundle_catalog import (
    BundleSkillCatalog,
    catalog_by_id,
    load_bundle_skills,
    recommended_skill_ids,
)
from potpie_context_engine.adapters.outbound.skills.claude_target import (
    ClaudeAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
    ProjectAgentTarget,
)
from potpie_context_engine.adapters.outbound.skills.template_resources import (
    MissingTemplateResources,
    NO_TEMPLATE_RESOURCES,
    PackageTemplateResources,
    TemplateResourceProvider,
    resolve_template_resources,
)

__all__ = [
    "BundleSkillCatalog",
    "ClaudeAgentTarget",
    "CodexAgentTarget",
    "CursorAgentTarget",
    "MissingTemplateResources",
    "NO_TEMPLATE_RESOURCES",
    "OpenCodeAgentTarget",
    "PackageTemplateResources",
    "ProjectAgentTarget",
    "TemplateResourceProvider",
    "catalog_by_id",
    "load_bundle_skills",
    "recommended_skill_ids",
    "resolve_template_resources",
]
