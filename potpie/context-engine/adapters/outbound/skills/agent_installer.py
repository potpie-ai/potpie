"""Install packaged AGENTS.md / CLAUDE.md and repo-local skills into a target repository."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

_CLAUDE_MARKER_RE = re.compile(
    r"<!-- (?:context-engine|potpie)-start -->.*?<!-- (?:context-engine|potpie)-end -->",
    re.DOTALL,
)

AGENT_TYPES = ("default", "codex", "claude", "cursor", "opencode")
_SOURCE_SKILLS_PREFIX = ".agents/skills/"


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
                stack.append((child, child_rel))
                continue
            out.append((child_rel, child.read_text(encoding="utf-8")))
    return sorted(out, key=lambda item: item[0].as_posix())


def iter_template_files() -> list[tuple[Path, str]]:
    """Return agent_bundle template files (default / codex path)."""
    return _iter_bundle_files("agent_bundle")


def _merge_claude_md(existing: str, section: str, *, force: bool) -> tuple[str, str]:
    """Return (merged_content, action) where action is 'unchanged'|'updated'|'created'."""
    if _CLAUDE_MARKER_RE.search(existing):
        merged = _CLAUDE_MARKER_RE.sub(section.strip(), existing)
        if merged == existing:
            return existing, "unchanged"
        if not force:
            return existing, "skipped"
        return merged, "updated"
    # No marker found — append the section
    separator = "\n\n" if existing.strip() else ""
    merged = existing.rstrip() + separator + section.strip() + "\n"
    return merged, "created"


def _remap_skills_path(rel_path: Path, target_prefix: str) -> Path | None:
    posix = rel_path.as_posix()
    if not posix.startswith(_SOURCE_SKILLS_PREFIX):
        return None
    return Path(target_prefix) / posix[len(_SOURCE_SKILLS_PREFIX) :]


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
) -> None:
    for rel_path, content in _iter_bundle_files(bundle_name):
        if include is not None and not include(rel_path):
            continue
        out_path = rel_path if remap is None else remap(rel_path)
        if out_path is None:
            continue
        target = install_root / out_path

        # Special handling: merge CLAUDE.md section instead of overwriting
        if out_path.name == "CLAUDE.md":
            section = content
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            merged, action = _merge_claude_md(existing, section, force=force)
            if action == "skipped":
                result.skipped.append(out_path.as_posix())
                continue
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


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    force: bool = False,
) -> InstallResult:
    """Install agent bundle files into the nearest git repo root under *path*.

    - ``default`` / ``codex``: ``AGENTS.md`` + ``.agents/skills/``
    - ``claude``: ``CLAUDE.md`` (+ ``.claude/`` when present in bundle)
    - ``cursor``: ``AGENTS.md`` + ``.cursor/skills/``
    - ``opencode``: ``.opencode/skills/``
    """
    root = resolve_install_root(path)
    result = InstallResult(root=str(root))

    normalized = agent.strip().lower() if agent else "default"
    if normalized not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type {agent!r}. Choose one of: {', '.join(AGENT_TYPES)}"
        )

    if normalized == "claude":
        _install_bundle(root, "claude_bundle", result, force=force)
    elif normalized == "cursor":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=_cursor_bundle_include,
            remap=_cursor_bundle_remap,
        )
    elif normalized == "opencode":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            force=force,
            include=lambda rel: rel.as_posix().startswith(_SOURCE_SKILLS_PREFIX),
            remap=_opencode_bundle_remap,
        )
    else:
        _install_bundle(root, "agent_bundle", result, force=force)

    return result
