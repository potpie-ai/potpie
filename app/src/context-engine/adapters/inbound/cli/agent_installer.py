"""Install packaged AGENTS.md / CLAUDE.md and repo-local skills into a target repository."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

_CLAUDE_MARKER_RE = re.compile(
    r"<!-- (?:context-engine|potpie)-start -->.*?<!-- (?:context-engine|potpie)-end -->",
    re.DOTALL,
)

AGENT_TYPES = ("default", "codex", "claude")


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


def _install_bundle(
    install_root: Path,
    bundle_name: str,
    result: InstallResult,
    *,
    force: bool,
) -> None:
    for rel_path, content in _iter_bundle_files(bundle_name):
        target = install_root / rel_path

        # Special handling: merge CLAUDE.md section instead of overwriting
        if rel_path.name == "CLAUDE.md":
            section = content
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            merged, action = _merge_claude_md(existing, section, force=force)
            if action == "skipped":
                result.skipped.append(rel_path.as_posix())
                continue
            if action == "unchanged":
                result.unchanged.append(rel_path.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(merged, encoding="utf-8")
            if action == "created":
                result.created.append(rel_path.as_posix())
            else:
                result.updated.append(rel_path.as_posix())
            continue

        # Regular file install
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if existing == content:
                result.unchanged.append(rel_path.as_posix())
                continue
            if not force:
                result.skipped.append(rel_path.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            result.updated.append(rel_path.as_posix())
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result.created.append(rel_path.as_posix())


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    force: bool = False,
) -> InstallResult:
    """Install agent bundle files into the nearest git repo root under *path*.

    agent="default" or "codex" installs the AGENTS.md + .agents/skills bundle.
    agent="claude" installs the CLAUDE.md section + .claude/commands bundle.
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
    else:
        _install_bundle(root, "agent_bundle", result, force=force)

    return result
