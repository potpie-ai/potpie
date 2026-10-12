"""Built-in skill catalog, read from the packaged skill bundle.

The bundled ``.agents/skills/*/SKILL.md`` files are the single source of truth
for skill content and metadata. This module turns their front matter into
:class:`SkillInfo` records for the Skill Manager (list/install/status/drift).
"""

from __future__ import annotations

from functools import lru_cache

from potpie.skills.bundle import (
    SKILL_MANIFEST,
    clear_bundle_file_cache,
    parse_front_matter,
    skill_files,
)
from potpie.skills.contracts import SkillInfo


def _title_from_body(body: str, *, skill_id: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return skill_id


def _is_recommended(meta: dict[str, str]) -> bool:
    """``recommended: false`` hides a skill; anything unrecognised keeps it."""
    return (meta.get("recommended") or "").strip().lower() not in {"false", "no", "0"}


@lru_cache(maxsize=1)
def load_bundle_skills() -> tuple[SkillInfo, ...]:
    """Every recommended skill in the bundle, sorted by id."""
    skills: list[SkillInfo] = []
    for skill_id, rel, raw in skill_files():
        if rel.as_posix() != SKILL_MANIFEST:
            continue
        meta, body = parse_front_matter(raw)
        if not _is_recommended(meta):
            continue
        skills.append(
            SkillInfo(
                id=skill_id,
                title=meta.get("title") or _title_from_body(body, skill_id=skill_id),
                version=meta.get("version", "1"),
                description=meta.get("description", ""),
            )
        )
    return tuple(sorted(skills, key=lambda skill: skill.id))


@lru_cache(maxsize=1)
def catalog_by_id() -> dict[str, SkillInfo]:
    return {skill.id: skill for skill in load_bundle_skills()}


@lru_cache(maxsize=1)
def recommended_skill_ids() -> tuple[str, ...]:
    return tuple(skill.id for skill in load_bundle_skills())


# Module-level tuples for callers that predate the functions above.
BUILTIN_SKILLS: tuple[SkillInfo, ...] = load_bundle_skills()
RECOMMENDED_SKILL_IDS: tuple[str, ...] = recommended_skill_ids()


def clear_bundle_catalog_cache() -> None:
    """Test helper: drop cached scans of the packaged bundle."""
    clear_bundle_file_cache()
    load_bundle_skills.cache_clear()
    catalog_by_id.cache_clear()
    recommended_skill_ids.cache_clear()


__all__ = [
    "BUILTIN_SKILLS",
    "RECOMMENDED_SKILL_IDS",
    "catalog_by_id",
    "clear_bundle_catalog_cache",
    "load_bundle_skills",
    "recommended_skill_ids",
]
