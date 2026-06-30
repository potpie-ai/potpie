"""Built-in skill catalog loaded from packaged ``agent_bundle`` templates.

The bundled ``.agents/skills/*/SKILL.md`` files are the single source of truth
for skill content and metadata. This module scans those templates at runtime and
builds :class:`SkillInfo` records for the Skill Manager (list/install/status/drift).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from domain.ports.services.skill_manager import SkillInfo

AGENT_BUNDLE_NAME = "agent_bundle"
SKILLS_PREFIX = ".agents/skills/"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Return (front-matter key/values, markdown body)."""
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        return {}, raw
    block = raw[4:end]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        meta[key.strip()] = _strip_yaml_scalar(value.strip())
    return meta, raw[end + 5 :]


def _strip_yaml_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _title_from_body(body: str, *, skill_id: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return skill_id


def _coerce_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"false", "no", "0"}:
        return False
    if normalized in {"true", "yes", "1"}:
        return True
    return default


def _skill_info(skill_id: str, raw: str) -> SkillInfo:
    meta, body = _parse_front_matter(raw)
    title = meta.get("title") or _title_from_body(body, skill_id=skill_id)
    description = meta.get("description", "")
    version = meta.get("version", "1")
    return SkillInfo(
        id=skill_id,
        title=title,
        version=version,
        description=description,
    )


def _is_recommended(meta: dict[str, str]) -> bool:
    return _coerce_bool(meta.get("recommended"), default=True)


@lru_cache(maxsize=1)
def load_bundle_skills() -> tuple[SkillInfo, ...]:
    """All skills shipped under ``agent_bundle/.agents/skills/``, sorted by id."""
    root = resources.files("adapters.inbound.cli").joinpath(
        "templates", AGENT_BUNDLE_NAME, SKILLS_PREFIX.rstrip("/")
    )
    skills: list[SkillInfo] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        skill_md = child.joinpath("SKILL.md")
        if not skill_md.is_file():
            continue
        raw = skill_md.read_text(encoding="utf-8")
        meta, _ = _parse_front_matter(raw)
        if not _is_recommended(meta):
            continue
        skills.append(_skill_info(child.name, raw))
    return tuple(skills)


@lru_cache(maxsize=1)
def catalog_by_id() -> dict[str, SkillInfo]:
    return {skill.id: skill for skill in load_bundle_skills()}


@lru_cache(maxsize=1)
def recommended_skill_ids() -> tuple[str, ...]:
    return tuple(skill.id for skill in load_bundle_skills())


# Backwards-compatible aliases for callers expecting module-level tuples.
BUILTIN_SKILLS: tuple[SkillInfo, ...] = load_bundle_skills()
RECOMMENDED_SKILL_IDS: tuple[str, ...] = recommended_skill_ids()


def clear_bundle_catalog_cache() -> None:
    """Test helper: drop cached scans of the packaged bundle."""
    load_bundle_skills.cache_clear()
    catalog_by_id.cache_clear()
    recommended_skill_ids.cache_clear()


__all__ = [
    "AGENT_BUNDLE_NAME",
    "BUILTIN_SKILLS",
    "RECOMMENDED_SKILL_IDS",
    "SKILLS_PREFIX",
    "catalog_by_id",
    "clear_bundle_catalog_cache",
    "load_bundle_skills",
    "recommended_skill_ids",
]
