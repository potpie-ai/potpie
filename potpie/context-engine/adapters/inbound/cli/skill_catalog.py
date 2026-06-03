"""Potpie skill catalog + filesystem helpers for `potpie skills`.

Contract:
- Canonical install dir: `.agents/skills/<skill-id>`
- Bundled catalog: packaged templates at `templates/agent_bundle/.agents/skills/<skill-id>/SKILL.md`
- Hashing: deterministic SHA-256 over sorted relative paths + bytes, ignoring:
  `.git/`, `__pycache__/`, `__pypackages__/`, and `metadata.json`.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CANONICAL_SKILLS_DIR = Path(".agents") / "skills"
LOCK_PATH = Path(".agents") / "skills-lock.json"
SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
IGNORED_HASH_PARTS = {".git", "__pycache__", "__pypackages__"}
IGNORED_HASH_FILES = {"metadata.json"}


class SkillCatalogError(ValueError):
    """Catalog or skill metadata validation failure."""


@dataclass(frozen=True)
class SkillEntry:
    """A first-party Potpie skill catalog entry."""

    id: str
    name: str
    description: str
    template_path: str  # repo-relative SKILL.md path
    template_hash: str  # sha256:...
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_catalog_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "templatePath": self.template_path,
            "templateHash": self.template_hash,
            "metadata": self.metadata,
        }


def validate_skill_id(skill_id: str) -> str:
    """Return a normalized skill id or raise ``SkillCatalogError``."""

    value = (skill_id or "").strip()
    if (
        not SKILL_ID_RE.match(value)
        or "/" in value
        or "\\" in value
        or ".." in value
        or value.startswith(".")
        or value.endswith(".")
        or value.startswith("-")
        or value.endswith("-")
    ):
        raise SkillCatalogError(f"Invalid skill id: {skill_id!r}")
    return value


def canonical_skills_dir(root: Path) -> Path:
    return root / CANONICAL_SKILLS_DIR


def lock_path(root: Path) -> Path:
    return root / LOCK_PATH


def skill_dir(root: Path, skill_id: str) -> Path:
    """Resolve a skill directory and assert it remains in the canonical dir."""

    safe_id = validate_skill_id(skill_id)
    base = canonical_skills_dir(root).resolve()
    target = (base / safe_id).resolve()
    if target != base / safe_id or not _is_relative_to(target, base):
        raise SkillCatalogError(f"Unsafe skill path for {skill_id!r}")
    return target


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_skill_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse simple YAML frontmatter without executing arbitrary tags.

    The MVP only needs scalar keys. Dotted keys such as ``metadata.internal`` are
    expanded into nested dictionaries for compatibility with reference skills.
    """

    if not text.startswith("---\n"):
        raise SkillCatalogError("SKILL.md is missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise SkillCatalogError("SKILL.md frontmatter is not closed")
    raw = text[4:end]
    body = text[end + len("\n---") :].lstrip("\n")
    data: dict[str, Any] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillCatalogError(f"Invalid frontmatter line: {line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        parsed = _parse_scalar(value.strip())
        _assign_dotted_key(data, key, parsed)
    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillCatalogError("SKILL.md frontmatter requires string name")
    if not isinstance(description, str) or not description.strip():
        raise SkillCatalogError("SKILL.md frontmatter requires string description")
    data["name"] = name.strip()
    data["description"] = description.strip()
    return data, body


def discover_bundled_skills() -> tuple[list[SkillEntry], list[dict[str, Any]]]:
    """Discover first-party bundled skills from packaged templates."""

    templates_root = _packaged_skills_root()
    entries: list[SkillEntry] = []
    diagnostics: list[dict[str, Any]] = []

    for child in sorted(templates_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        try:
            skill_id = validate_skill_id(child.name)
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            metadata, _ = parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
            if _is_internal(metadata):
                continue
            entries.append(
                SkillEntry(
                    id=skill_id,
                    name=str(metadata["name"]),
                    description=str(metadata["description"]),
                    template_path=f"templates/agent_bundle/.agents/skills/{skill_id}/SKILL.md",
                    template_hash=hash_traversable_dir(child),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in {"name", "description"}
                    },
                )
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "code": "INVALID_CATALOG_SKILL",
                    "skillId": child.name,
                    "message": str(exc),
                }
            )
    return sorted(entries, key=lambda entry: entry.id), diagnostics


def discover_installed_skills(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read installed repo-local skills from ``.agents/skills``."""

    skills_root = canonical_skills_dir(root)
    if not skills_root.exists():
        return [], []
    installed: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for child in sorted(skills_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        try:
            skill_id = validate_skill_id(child.name)
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            metadata, _ = parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
            installed.append(
                {
                    "id": skill_id,
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "installedPath": relative_to_root(skill_md, root),
                    "installedHash": hash_path_dir(child),
                    "metadata": {
                        k: v
                        for k, v in metadata.items()
                        if k not in {"name", "description"}
                    },
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "code": "INVALID_INSTALLED_SKILL",
                    "skillId": child.name,
                    "message": str(exc),
                }
            )
    return installed, diagnostics


def copy_bundled_skill_atomic(skill_id: str, root: Path) -> str:
    """Copy one bundled (packaged) skill into the canonical project directory atomically."""

    safe_id = validate_skill_id(skill_id)
    source = _packaged_skills_root().joinpath(safe_id)
    if not source.is_dir() or not source.joinpath("SKILL.md").is_file():
        raise SkillCatalogError(f"Unknown bundled skill: {skill_id!r}")
    target = skill_dir(root, safe_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(prefix=f".{safe_id}.tmp-", dir=str(target.parent.resolve()))
    )
    try:
        _copy_traversable_dir(source, tmp)
        installed_hash = hash_path_dir(tmp)
        if target.exists():
            shutil.rmtree(target)
        tmp.replace(target)
        return installed_hash
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def hash_path_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in _iter_hashable_paths(path):
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def hash_traversable_dir(path: Traversable) -> str:
    digest = hashlib.sha256()
    for rel, file_obj in _iter_hashable_traversables(path):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_obj.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def packaged_skill_hash(skill_id: str) -> str:
    return hash_traversable_dir(
        _packaged_skills_root().joinpath(validate_skill_id(skill_id))
    )


def _packaged_skills_root() -> Traversable:
    return resources.files("adapters.inbound.cli").joinpath(
        "templates", "agent_bundle", ".agents", "skills"
    )


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _assign_dotted_key(data: dict[str, Any], key: str, value: Any) -> None:
    if "." not in key:
        data[key] = value
        return
    current = data
    parts = [part.strip() for part in key.split(".") if part.strip()]
    for part in parts[:-1]:
        nested = current.setdefault(part, {})
        if not isinstance(nested, dict):
            raise SkillCatalogError(f"Conflicting frontmatter key: {key!r}")
        current = nested
    current[parts[-1]] = value


def _is_internal(metadata: dict[str, Any]) -> bool:
    internal = False
    meta = metadata.get("metadata")
    if isinstance(meta, dict):
        internal = meta.get("internal") is True
    if not internal:
        return False
    return os.getenv("INSTALL_INTERNAL_SKILLS", "").strip().lower() not in {
        "1",
        "true",
    }


def _iter_hashable_paths(path: Path) -> list[Path]:
    files: list[Path] = []
    for file_path in path.rglob("*"):
        rel_parts = set(file_path.relative_to(path).parts)
        if rel_parts & IGNORED_HASH_PARTS:
            continue
        if file_path.name in IGNORED_HASH_FILES:
            continue
        if file_path.is_file() and not file_path.is_symlink():
            files.append(file_path)
    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def _iter_hashable_traversables(
    root: Traversable, prefix: Path = Path(".")
) -> list[tuple[str, Traversable]]:
    files: list[tuple[str, Traversable]] = []
    for child in root.iterdir():
        rel = prefix / child.name
        if set(rel.parts) & IGNORED_HASH_PARTS or child.name in IGNORED_HASH_FILES:
            continue
        if child.is_dir():
            files.extend(_iter_hashable_traversables(child, rel))
        elif child.is_file():
            files.append((rel.as_posix(), child))
    return sorted(files, key=lambda item: item[0])


def _copy_traversable_dir(source: Traversable, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        dest = target / child.name
        if child.is_dir():
            _copy_traversable_dir(child, dest)
        elif child.is_file():
            dest.write_bytes(child.read_bytes())


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
