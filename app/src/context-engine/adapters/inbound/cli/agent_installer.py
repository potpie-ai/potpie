"""Install packaged AGENTS.md and repo-local skills into a target repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path


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


def iter_template_files() -> list[tuple[Path, str]]:
    """Return packaged template files as repo-relative paths and UTF-8 text."""
    root = resources.files("adapters.inbound.cli").joinpath("templates", "agent_bundle")
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


def install_agent_bundle(path: str | Path = ".", *, force: bool = False) -> InstallResult:
    root = resolve_install_root(path)
    result = InstallResult(root=str(root))

    for rel_path, content in iter_template_files():
        target = root / rel_path
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

    return result
