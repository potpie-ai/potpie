"""Install packaged AGENTS.md / CLAUDE.md and skills into agent targets."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable

_MANAGED_MARKER_RE = re.compile(
    r"<!-- (?:context-engine|potpie)-start -->.*?<!-- (?:context-engine|potpie)-end -->",
    re.DOTALL,
)
_DEFAULT_MERGE_FILES = frozenset({"AGENTS.md", "CLAUDE.md"})

AGENT_TYPES = ("default", "codex", "claude", "claude-plugin", "cursor", "opencode")
_SOURCE_SKILLS_PREFIX = ".agents/skills/"
# The Claude Code plugin installs as a self-contained directory so its
# ``.claude-plugin/plugin.json`` stays the plugin root for ``/plugin marketplace add``.
_CLAUDE_PLUGIN_PREFIX = ".claude/potpie-plugin"


@dataclass
class InstallResult:
    root: str
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ok"] = True
        return data


def resolve_install_root(path: str | Path) -> Path:
    """Prefer the nearest git repo root; otherwise install into the given path."""
    target = Path(path).resolve()
    if target.is_file():
        raise ValueError(f"Expected a directory path, got file: {target}")
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return target


def _iter_bundle_files(bundle_name: str) -> list[tuple[Path, str]]:
    """Return packaged template files from the named bundle as (repo-relative path, UTF-8 text)."""
    root = resources.files("adapters.inbound.cli").joinpath("templates", bundle_name)
    out: list[tuple[Path, str]] = []
    stack = [(root, Path("."))]
    while stack:
        current, rel = stack.pop()
        for child in current.iterdir():
            child_rel = rel / child.name
            if child.is_dir():
                # Never install compiled-bytecode caches that may sit beside the
                # template sources (e.g. a stray ``__pycache__`` from a test run).
                if child.name == "__pycache__":
                    continue
                stack.append((child, child_rel))
                continue
            if child.name.endswith((".pyc", ".pyo")):
                continue
            out.append((child_rel, child.read_text(encoding="utf-8")))
    return sorted(out, key=lambda item: item[0].as_posix())


def iter_template_files() -> list[tuple[Path, str]]:
    """Return agent_bundle template files (default / codex path)."""
    return _iter_bundle_files("agent_bundle")


def _merge_managed_markdown(existing: str, section: str) -> tuple[str, str]:
    """Return (merged_content, action) where action is 'unchanged'|'updated'|'created'."""
    normalized_section = section.strip()
    unmarked_section = _strip_managed_markers(normalized_section)
    if _MANAGED_MARKER_RE.search(existing):
        merged = _MANAGED_MARKER_RE.sub(normalized_section, existing)
        if merged == existing:
            return existing, "unchanged"
        return merged, "updated"
    if existing.strip() == unmarked_section.strip():
        merged = normalized_section + "\n"
        if merged == existing:
            return existing, "unchanged"
        return merged, "updated"
    if unmarked_section in existing:
        merged = existing.replace(unmarked_section, normalized_section, 1)
        if merged == existing:
            return existing, "unchanged"
        return merged, "updated"
    # No marker found — append the section
    separator = "\n\n" if existing.strip() else ""
    merged = existing.rstrip() + separator + normalized_section + "\n"
    action = "updated" if existing.strip() else "created"
    return merged, action


def _strip_managed_markers(section: str) -> str:
    lines = section.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().endswith("-start -->"):
        lines = lines[1:]
    if lines and lines[-1].strip().endswith("-end -->"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _remap_skills_path(rel_path: Path, target_prefix: str) -> Path | None:
    posix = rel_path.as_posix()
    if not posix.startswith(_SOURCE_SKILLS_PREFIX):
        return None
    return Path(target_prefix) / posix[len(_SOURCE_SKILLS_PREFIX) :]


def _skill_id_for_path(rel_path: Path) -> str | None:
    """Return the bundled skill id for a template path, if it is under .agents/skills."""
    posix = rel_path.as_posix()
    if not posix.startswith(_SOURCE_SKILLS_PREFIX):
        return None
    rest = posix[len(_SOURCE_SKILLS_PREFIX) :]
    return rest.split("/", 1)[0] if rest else None


def _normalize_skill_ids(skill_ids: Iterable[str] | None) -> frozenset[str] | None:
    if skill_ids is None:
        return None
    return frozenset(sid.strip() for sid in skill_ids if sid and sid.strip())


def _include_selected_skills(
    rel_path: Path, selected: frozenset[str] | None
) -> bool:
    sid = _skill_id_for_path(rel_path)
    return sid is not None and (selected is None or sid in selected)


def _install_file(
    install_root: Path,
    rel_path: Path,
    content: str,
    result: InstallResult,
    *,
    force: bool,
) -> None:
    target = install_root / rel_path
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing == content:
            result.unchanged.append(rel_path.as_posix())
            return
        if not force:
            result.skipped.append(rel_path.as_posix())
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result.updated.append(rel_path.as_posix())
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    result.created.append(rel_path.as_posix())


def _install_bundle(
    install_root: Path,
    bundle_name: str,
    result: InstallResult,
    *,
    force: bool,
    include: Callable[[Path], bool] | None = None,
    remap: Callable[[Path], Path | None] | None = None,
    merge_files: frozenset[str] = _DEFAULT_MERGE_FILES,
) -> None:
    for rel_path, content in _iter_bundle_files(bundle_name):
        if include is not None and not include(rel_path):
            continue
        out_path = rel_path if remap is None else remap(rel_path)
        if out_path is None:
            continue
        target = install_root / out_path

        # Special handling: merge managed markdown sections instead of overwriting
        # the whole user-authored file.
        if out_path.name in merge_files:
            section = content
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            merged, action = _merge_managed_markdown(existing, section)
            if action == "unchanged":
                result.unchanged.append(out_path.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(merged, encoding="utf-8")
            if action == "created":
                result.created.append(out_path.as_posix())
            else:
                result.updated.append(out_path.as_posix())
            continue

        _install_file(install_root, out_path, content, result, force=force)


def _cursor_bundle_include(rel_path: Path) -> bool:
    posix = rel_path.as_posix()
    return posix == "AGENTS.md" or posix.startswith(_SOURCE_SKILLS_PREFIX)


def _cursor_bundle_remap(rel_path: Path) -> Path | None:
    remapped = _remap_skills_path(rel_path, ".cursor/skills")
    if remapped is not None:
        return remapped
    if rel_path.as_posix() == "AGENTS.md":
        return rel_path
    return None


def _opencode_bundle_remap(rel_path: Path) -> Path | None:
    return _remap_skills_path(rel_path, ".opencode/skills")


def _claude_skills_bundle_remap(rel_path: Path) -> Path | None:
    return _remap_skills_path(rel_path, ".claude/skills")


def _claude_plugin_remap(rel_path: Path) -> Path | None:
    # Install the whole plugin under one directory, preserving its internal layout.
    return Path(_CLAUDE_PLUGIN_PREFIX) / rel_path


def install_skill_bundle(
    skills_root: str | Path,
    *,
    skill_ids: Iterable[str] | None = None,
    force: bool = False,
) -> InstallResult:
    """Install selected packaged skills directly into a skills root.

    ``skills_root`` is the directory that contains one subdirectory per skill,
    for example ``~/.cursor/skills`` or ``~/.agents/skills``.
    """
    root = Path(skills_root).expanduser().resolve()
    result = InstallResult(root=str(root))
    selected = _normalize_skill_ids(skill_ids)
    _install_bundle(
        root,
        "agent_bundle",
        result,
        force=force,
        include=lambda rel: _include_selected_skills(rel, selected),
        remap=lambda rel: Path(rel.as_posix()[len(_SOURCE_SKILLS_PREFIX) :]),
    )
    return result


def install_global_agent_instructions(
    root: str | Path,
    *,
    agent: str = "default",
    force: bool = True,
) -> InstallResult:
    """Install compact global instructions for harnesses with file-based rules.

    The project bundle is intentionally detailed. This global bundle stays tiny
    because it can be loaded into every prompt across repositories.
    """
    install_root = Path(root).expanduser().resolve()
    result = InstallResult(root=str(install_root))
    normalized = agent.strip().lower() if agent else "default"
    if normalized == "claude":
        filename = "CLAUDE.md"
    elif normalized in {"default", "codex"}:
        filename = "AGENTS.md"
    else:
        return result

    _install_bundle(
        install_root,
        "global_agent_bundle",
        result,
        force=force,
        include=lambda rel: rel.as_posix() == filename,
        merge_files=frozenset({filename}),
    )
    return result


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    force: bool = False,
    skill_ids: Iterable[str] | None = None,
) -> InstallResult:
    """Install agent bundle files into the nearest git repo root under *path*.

    - ``default`` / ``codex``: ``AGENTS.md`` + ``.agents/skills/``
    - ``claude``: ``CLAUDE.md`` (+ ``.claude/`` when present in bundle)
    - ``claude-plugin``: the Claude Code plugin under ``.claude/potpie-plugin/``
    - ``cursor``: ``AGENTS.md`` + ``.cursor/skills/``
    - ``opencode``: ``.opencode/skills/``
    """
    root = resolve_install_root(path)
    result = InstallResult(root=str(root))
    selected = _normalize_skill_ids(skill_ids)

    normalized = agent.strip().lower() if agent else "default"
    if normalized not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type {agent!r}. Choose one of: {', '.join(AGENT_TYPES)}"
        )

    if normalized == "claude":
        _install_bundle(root, "claude_bundle", result, force=force)
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=lambda rel: _include_selected_skills(rel, selected),
            remap=_claude_skills_bundle_remap,
        )
    elif normalized == "claude-plugin":
        _install_bundle(
            root,
            "claude_plugin",
            result,
            force=force,
            remap=_claude_plugin_remap,
        )
    elif normalized == "cursor":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=lambda rel: rel.as_posix() == "AGENTS.md"
            or _include_selected_skills(rel, selected),
            remap=_cursor_bundle_remap,
        )
    elif normalized == "opencode":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=lambda rel: _include_selected_skills(rel, selected),
            remap=_opencode_bundle_remap,
        )
    else:
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=lambda rel: rel.as_posix() == "AGENTS.md"
            or _include_selected_skills(rel, selected),
        )

    return result


def project_skill_path(root: str | Path, *, agent: str, skill_id: str) -> Path:
    """Return the project-scope SKILL.md path for a harness and skill id."""
    install_root = resolve_install_root(root)
    normalized = agent.strip().lower() if agent else "default"
    if normalized == "cursor":
        return install_root / ".cursor" / "skills" / skill_id / "SKILL.md"
    if normalized == "claude":
        return install_root / ".claude" / "skills" / skill_id / "SKILL.md"
    if normalized == "opencode":
        return install_root / ".opencode" / "skills" / skill_id / "SKILL.md"
    return install_root / ".agents" / "skills" / skill_id / "SKILL.md"
