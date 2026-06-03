"""Built-in skill catalog — the recommended skills the CLI ships.

A static catalog of the skills that teach an agent harness to use the
``potpie`` CLI and the four context tools. Mirrors the bundles under
``adapters/inbound/cli/templates/``.

    TODO(stage-N): source versions/manifests from the template bundles and the
    optional cloud catalog instead of this hand-written list.
"""

from __future__ import annotations

from domain.ports.services.skill_manager import SkillInfo

BUILTIN_SKILLS: tuple[SkillInfo, ...] = (
    SkillInfo(
        id="potpie-cli",
        title="Potpie CLI",
        version="1",
        description="How to drive the potpie CLI (setup, status, resolve, record).",
    ),
    SkillInfo(
        id="potpie-agent-context",
        title="Agent Context",
        version="1",
        description="How to use the four context tools (resolve/search/record/status).",
    ),
    SkillInfo(
        id="potpie-pot-scope",
        title="Pot Scope",
        version="1",
        description="Working within a pot boundary and scoping requests.",
    ),
    SkillInfo(
        id="potpie-cli-troubleshooting",
        title="CLI Troubleshooting",
        version="1",
        description="Diagnosing daemon/backend/source issues from the CLI.",
    ),
)

RECOMMENDED_SKILL_IDS: tuple[str, ...] = tuple(s.id for s in BUILTIN_SKILLS)


def catalog_by_id() -> dict[str, SkillInfo]:
    return {s.id: s for s in BUILTIN_SKILLS}


__all__ = ["BUILTIN_SKILLS", "RECOMMENDED_SKILL_IDS", "catalog_by_id"]
