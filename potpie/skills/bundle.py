"""Read the packaged templates: one skill bundle and one routing block.

``templates/agent_bundle/.agents/skills/<id>/`` holds every skill; each
harness installs from it, remapped to its own layout.
``templates/routing/POTPIE.md`` is the instruction block merged into
``AGENTS.md`` / ``CLAUDE.md``. The catalog and the installer both read the
bundle through this module, so they agree on what a skill is.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

AGENT_BUNDLE = "agent_bundle"
ROUTING_BUNDLE = "routing"
ROUTING_FILE = "POTPIE.md"
#: Where skills sit inside the agent bundle.
SKILLS_PREFIX = ".agents/skills/"
SKILL_MANIFEST = "SKILL.md"


@lru_cache(maxsize=8)
def bundle_files(bundle_name: str) -> tuple[tuple[Path, str], ...]:
    """The named bundle's files as ``(bundle-relative path, UTF-8 text)``, sorted.

    Cached because it is on a read path: every content-drift check dry-runs an
    install against it, and ``skills status`` runs one per recommended skill.
    The bundle ships inside the installed wheel and cannot change under a
    running process; tests that edit templates call
    :func:`clear_bundle_file_cache`.
    """
    root = resources.files("potpie.cli").joinpath("templates", bundle_name)
    out: list[tuple[Path, str]] = []
    stack = [(root, Path("."))]
    while stack:
        current, rel = stack.pop()
        for child in current.iterdir():
            child_rel = rel / child.name
            if child.is_dir():
                # Never ship a bytecode cache a test run left beside the sources.
                if child.name != "__pycache__":
                    stack.append((child, child_rel))
                continue
            if not child.name.endswith((".pyc", ".pyo")):
                out.append((child_rel, child.read_text(encoding="utf-8")))
    return tuple(sorted(out, key=lambda item: item[0].as_posix()))


def clear_bundle_file_cache() -> None:
    """Test helper: drop cached reads of the packaged bundles."""
    bundle_files.cache_clear()


def iter_template_files() -> tuple[tuple[Path, str], ...]:
    """Every file in the agent bundle, at its bundle-relative path."""
    return bundle_files(AGENT_BUNDLE)


def _skill_id_for_path(rel_path: Path) -> str | None:
    """The skill a bundle path belongs to, or ``None`` outside ``.agents/skills``."""
    posix = rel_path.as_posix()
    if not posix.startswith(SKILLS_PREFIX):
        return None
    return posix[len(SKILLS_PREFIX) :].split("/", 1)[0] or None


def skill_files() -> tuple[tuple[str, Path, str], ...]:
    """``(skill id, path inside the skill directory, text)`` for every skill file."""
    out: list[tuple[str, Path, str]] = []
    for rel_path, content in iter_template_files():
        skill_id = _skill_id_for_path(rel_path)
        if skill_id is not None:
            out.append(
                (skill_id, rel_path.relative_to(SKILLS_PREFIX + skill_id), content)
            )
    return tuple(out)


def bundle_skill_ids() -> frozenset[str]:
    """Ids of every skill the bundle carries a ``SKILL.md`` for."""
    return frozenset(
        skill_id
        for skill_id, rel, _ in skill_files()
        if rel.as_posix() == SKILL_MANIFEST
    )


def routing_block() -> str:
    """The managed instruction block, markers included."""
    return dict(bundle_files(ROUTING_BUNDLE))[Path(ROUTING_FILE)]


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """``(front-matter key/values, markdown body)`` of one template."""
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        return {}, raw
    meta: dict[str, str] = {}
    for line in raw[4:end].splitlines():
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


__all__ = [
    "AGENT_BUNDLE",
    "ROUTING_BUNDLE",
    "ROUTING_FILE",
    "SKILLS_PREFIX",
    "SKILL_MANIFEST",
    "bundle_files",
    "bundle_skill_ids",
    "clear_bundle_file_cache",
    "iter_template_files",
    "parse_front_matter",
    "routing_block",
    "skill_files",
]
