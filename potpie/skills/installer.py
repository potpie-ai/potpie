"""Write packaged skills and the routing block to disk, and take them back out.

One install and one uninstall. Callers say *where* (a root, the skills
directory under it, the instruction file under it) and *what* (which skills,
whether to write the routing block); :mod:`potpie.skills.harnesses` says where
each harness wants them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Iterable

from potpie.skills.bundle import (
    bundle_skill_ids,
    iter_template_files,
    routing_block,
    skill_files,
)
from potpie.skills.errors import (
    InvalidSkillsInstallPathError,
    UnwritableSkillsTargetError,
)
from potpie.skills.harnesses import harness_layout

_MANAGED_MARKER_RE = re.compile(
    r"<!-- (?:context-engine|potpie)-start -->.*?<!-- (?:context-engine|potpie)-end -->",
    re.DOTALL,
)

# Earlier releases wrote two slash commands into a repository's
# ``.claude/commands/`` and, on request, a Claude Code plugin under
# ``.claude/potpie-plugin/``. Neither ships any more. A command file is Potpie's
# to delete only while it is byte-for-byte one of the versions Potpie wrote
# (SHA-256 with LF line endings); an edited copy is the user's.
_RETIRED_CLAUDE_COMMANDS_DIR = Path(".claude/commands")
_RETIRED_CLAUDE_COMMANDS: dict[str, frozenset[str]] = {
    "potpie-feature.md": frozenset(
        {
            "37376dd13143b3606d5f2c8c6ad66a83d084ca8e8bd48b7d14dda709187367e8",
            "42ec892a825ba2a026c21709ff543777bf5e5c67c6886762daec1980ecbb8d5f",
            "45475ab13ab3f9c94625b5a3be35aadbacb2e7656a82e93a7a747b6828f0d385",
            "667aebfb2e23f9c3797e61441bc2294bfcc7fb406ceb80f4e321e11a34b6d43c",
            "80f7ebf684c2124fe686312460591ded0ff455495c0cf114a121f70d4e40a468",
            "fa0dc815b5e330d64d9bb909af3db117cde32205f410b667ecfc314d85ccc1d8",
        }
    ),
    "potpie-record.md": frozenset(
        {
            "20b01220be5369e3ac63a8930ce814c0f223ccc0d4a7c897f346275d0b8e2e93",
            "2d8380e61c141c4244d08defd7bc77b0660461050c3bbec16d08a22a1f749ea7",
            "598dece2a755203bb7b449c499d4ac2a0ecede9311936d57e286b11a62bb71bf",
            "8078b147c6be8e4b3c7c2b3cfc6adb014336efdcb8c81f2946ead8cd35570526",
            "8ec9c00249a67875298d3a55267013096be2cc17086a986f954599004d0f77e2",
            "c2f57c4822f92a2e01499999a0a831e2d1b2f4523b7b1ff0c313605ce840ccbf",
        }
    ),
}
_RETIRED_CLAUDE_PLUGIN_DIR = Path(".claude/potpie-plugin")


@dataclass
class InstallResult:
    root: str
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # Files an earlier release installed that a sweep deleted because they were
    # still exactly as shipped, and the ones it left for the user to decide on.
    removed: list[str] = field(default_factory=list)
    leftovers: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ok"] = True
        return data


@dataclass
class UninstallResult:
    """What an uninstall took back out — the mirror of ``InstallResult``.

    Its own shape rather than a reused ``InstallResult``: "created/updated" have
    no meaning for a removal, and a caller that reported one as the other would
    tell the user a file was written when it was deleted.
    """

    root: str
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    leftovers: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ok"] = True
        return data


def _unwritable_target_error(
    target: Path, exc: OSError, *, verb: str = "write"
) -> UnwritableSkillsTargetError:
    """The refusal owed to a caller whose install target cannot be written.

    A read-only repository (or a ``--path`` under someone else's ownership)
    surfaced as a bare ``PermissionError``, which the CLI reports as an
    unexpected internal error. The actual repair is one ``chmod`` on a
    directory that message never named, so the refusal names it.
    """
    blocked = _nearest_existing_dir(target)
    return UnwritableSkillsTargetError(
        f"Cannot {verb} {target}: {exc.strerror or exc}. "
        f"The harness directory is not writable, so nothing was changed there.",
        recommended_next_action=(
            f"make '{blocked}' writable, or point somewhere else with '--path <dir>'"
        ),
    )


def _nearest_existing_dir(target: Path) -> Path:
    """The closest ancestor that actually exists — the one to fix permissions on.

    Naming ``target.parent`` sent the operator to ``chmod`` a directory the
    failed ``mkdir`` never created; the unwritable one is always further up.
    """
    for candidate in target.parents:
        if candidate.is_dir():
            return candidate
    return target.parent


def _write_installed_file(target: Path, content: str) -> None:
    """Write one bundle file, translating an unwritable target into a refusal."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise _unwritable_target_error(target, exc) from exc


def _remove_installed_file(target: Path) -> None:
    try:
        target.unlink()
    except OSError as exc:
        raise _unwritable_target_error(target, exc, verb="remove") from exc


def resolve_install_root(path: str | Path) -> Path:
    """Prefer the nearest git repo root; otherwise install into the given path."""
    target = Path(path).resolve()
    if target.is_file():
        raise InvalidSkillsInstallPathError(
            f"Expected a directory path, got file: {target}"
        )
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return target


def prune_empty_dirs(directory: Path, *, stop_at: Path) -> None:
    """Drop directories a removal just emptied, up to but excluding ``stop_at``.

    Stops at the first non-empty parent, so a ``.claude/`` that still holds
    skills — or anything the user put there — survives. Without it a full
    uninstall left the shape of the install behind: an empty
    ``.claude/skills/`` that reads, to anyone who opens the repo, as an install
    that is still there.
    """
    root = stop_at.resolve()
    current = directory.resolve()
    while current != root and current.is_relative_to(root) and current.is_dir():
        try:
            if any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


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


def _strip_managed_section(existing: str) -> str:
    """Return *existing* with Potpie's managed block taken back out.

    The install side merges the block into a file the user also writes in, so
    the removal side has to be just as careful: what comes out is the marked
    section and nothing else. An empty string means the file held only Potpie's
    block and the caller should delete it rather than leave a husk behind.
    """
    if not _MANAGED_MARKER_RE.search(existing):
        return existing
    remainder = _MANAGED_MARKER_RE.sub("", existing).strip()
    return f"{remainder}\n" if remainder else ""


def _strip_managed_markers(section: str) -> str:
    lines = section.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().endswith("-start -->"):
        lines = lines[1:]
    if lines and lines[-1].strip().endswith("-end -->"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _selected(skill_ids: Iterable[str] | None) -> frozenset[str] | None:
    if skill_ids is None:
        return None
    return frozenset(sid.strip() for sid in skill_ids if sid and sid.strip())


def _install_file(
    install_root: Path,
    rel_path: Path,
    content: str,
    result: InstallResult,
    *,
    force: bool,
    dry_run: bool,
) -> None:
    target = install_root / rel_path
    if target.exists():
        if target.read_text(encoding="utf-8") == content:
            result.unchanged.append(rel_path.as_posix())
            return
        if not force:
            result.skipped.append(rel_path.as_posix())
            return
        if not dry_run:
            _write_installed_file(target, content)
        result.updated.append(rel_path.as_posix())
        return
    if not dry_run:
        _write_installed_file(target, content)
    result.created.append(rel_path.as_posix())


def _merge_routing_block(
    install_root: Path, rel_path: Path, result: InstallResult, *, dry_run: bool
) -> None:
    """Merge the managed block into an instruction file the user also writes in."""
    target = install_root / rel_path
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    merged, action = _merge_managed_markdown(existing, routing_block())
    if action == "unchanged":
        result.unchanged.append(rel_path.as_posix())
        return
    if not dry_run:
        _write_installed_file(target, merged)
    (result.created if action == "created" else result.updated).append(
        rel_path.as_posix()
    )


def _strip_routing_block(
    install_root: Path, rel_path: Path, result: UninstallResult, *, dry_run: bool
) -> None:
    """Take the managed block back out, keeping whatever else the user wrote.

    Deleting a hand-written ``CLAUDE.md`` because Potpie once appended to it
    would be a far bigger removal than the one the caller asked for; only a
    file that held nothing but the block is deleted.
    """
    target = install_root / rel_path
    if not target.exists():
        result.unchanged.append(rel_path.as_posix())
        return
    existing = target.read_text(encoding="utf-8")
    remainder = _strip_managed_section(existing)
    if remainder == existing:
        result.unchanged.append(rel_path.as_posix())
        return
    if not dry_run:
        if remainder:
            _write_installed_file(target, remainder)
        else:
            _remove_installed_file(target)
            prune_empty_dirs(target.parent, stop_at=install_root)
    result.removed.append(rel_path.as_posix())


def install_bundle(
    root: str | Path,
    *,
    skills_dir: str | PurePath = ".",
    skill_ids: Iterable[str] | None = None,
    instructions: str | PurePath | None = None,
    sweep_retired_claude_files: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> InstallResult:
    """Install packaged skills and/or the routing block under ``root``.

    - ``skill_ids``: which skills — ``None`` for every bundled skill, an empty
      collection for none. Each lands in ``<root>/<skills_dir>/<id>/``.
    - ``instructions``: the instruction file, relative to ``root``, to merge
      the routing block into; ``None`` leaves instruction files alone. The
      block is merged between markers, never written over the user's text.
    - ``sweep_retired_claude_files``: also clear what earlier releases put in
      a repository's ``.claude/`` (see :func:`_sweep_retired_claude_files`).

    Result paths are relative to ``root``. ``dry_run`` classifies every file
    exactly as a real run would — created / updated / unchanged / skipped — and
    writes nothing, which is what makes content drift detectable: the
    comparison is the installer's own, so it cannot drift from what install
    actually does.
    """
    install_root = Path(root).expanduser().resolve()
    result = InstallResult(root=str(install_root))
    if instructions is not None:
        _merge_routing_block(install_root, Path(instructions), result, dry_run=dry_run)
    selected = _selected(skill_ids)
    for skill_id, rel, content in skill_files():
        if selected is None or skill_id in selected:
            _install_file(
                install_root,
                Path(skills_dir) / skill_id / rel,
                content,
                result,
                force=force,
                dry_run=dry_run,
            )
    if sweep_retired_claude_files:
        removed, leftovers = _sweep_retired_claude_files(install_root, dry_run=dry_run)
        result.removed.extend(removed)
        result.leftovers.extend(leftovers)
    return result


def uninstall_bundle(
    root: str | Path,
    *,
    skills_dir: str | PurePath = ".",
    skill_ids: Iterable[str] | None = (),
    instructions: str | PurePath | None = None,
    sweep_retired_claude_files: bool = False,
    dry_run: bool = False,
) -> UninstallResult:
    """Take back what :func:`install_bundle` wrote under ``root``.

    The same parameters, so the removal side owns exactly what the install side
    wrote: ``skill_ids`` (``None`` for every bundled skill) deletes those skill
    directories whole, and ``instructions`` strips the managed block from that
    file. Leaving the block behind is why a harness kept routing to Potpie
    skills after ``skills remove --all`` had removed every one of them.
    """
    install_root = Path(root).expanduser().resolve()
    result = UninstallResult(root=str(install_root))
    if instructions is not None:
        _strip_routing_block(install_root, Path(instructions), result, dry_run=dry_run)
    selected = _selected(skill_ids)
    for skill_id in sorted(bundle_skill_ids() if selected is None else selected):
        skill_dir = install_root / skills_dir / skill_id
        if skill_dir.exists():
            result.removed.append((Path(skills_dir) / skill_id).as_posix())
        if not dry_run:
            shutil.rmtree(skill_dir, ignore_errors=True)
            # The skills directory is Potpie's own; once the last skill leaves
            # it, an empty `.claude/skills/` still reads as an install.
            prune_empty_dirs(skill_dir.parent, stop_at=install_root)
    if sweep_retired_claude_files:
        removed, leftovers = _sweep_retired_claude_files(install_root, dry_run=dry_run)
        result.removed.extend(removed)
        result.leftovers.extend(leftovers)
    return result


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    force: bool = False,
    skill_ids: Iterable[str] | None = None,
    instructions: bool = True,
    dry_run: bool = False,
    support_files: bool | None = None,
) -> InstallResult:
    """Install one harness's project bundle into the repo root containing ``path``.

    The entry point embedding hosts call: ``agent`` picks the project layout
    from :data:`~potpie.skills.harnesses.HARNESS_LAYOUTS` (``default`` is
    codex's ``AGENTS.md`` + ``.agents/skills``), and ``instructions=False``
    installs only the selected skills. ``support_files`` is the old name of
    ``instructions`` and wins when passed.
    """
    if support_files is not None:
        instructions = support_files
    root = resolve_install_root(path)
    layout = harness_layout(agent)
    return install_bundle(
        root,
        skills_dir=layout.project_skills,
        skill_ids=skill_ids,
        instructions=layout.project_instructions if instructions else None,
        sweep_retired_claude_files=instructions and layout.sweeps_retired_claude_files,
        force=force,
        dry_run=dry_run,
    )


def _sweep_retired_claude_files(
    install_root: Path, *, dry_run: bool
) -> tuple[list[str], list[dict[str, str]]]:
    """Clear what earlier releases installed for Claude and this one no longer ships.

    Returns ``(removed, leftovers)``. A retired slash command is deleted only
    when its bytes are a version Potpie wrote; anything else — an edited command,
    the plugin directory Claude Code may still have registered — is reported
    with the step that clears it, never deleted.
    """
    removed: list[str] = []
    leftovers: list[dict[str, str]] = []
    for name, shipped in _RETIRED_CLAUDE_COMMANDS.items():
        rel = _RETIRED_CLAUDE_COMMANDS_DIR / name
        target = install_root / rel
        if not target.is_file():
            continue
        if _lf_sha256(target) in shipped:
            if not dry_run:
                _remove_installed_file(target)
                prune_empty_dirs(target.parent, stop_at=install_root)
            removed.append(rel.as_posix())
            continue
        leftovers.append(
            {
                "path": rel.as_posix(),
                "recommended_next_action": (
                    f"Potpie no longer ships the /{Path(name).stem} command and "
                    "this copy was edited after install; delete it if you no "
                    "longer use it."
                ),
            }
        )
    if _is_retired_claude_plugin(install_root / _RETIRED_CLAUDE_PLUGIN_DIR):
        leftovers.append(
            {
                "path": _RETIRED_CLAUDE_PLUGIN_DIR.as_posix(),
                "recommended_next_action": (
                    "Potpie no longer ships a Claude Code plugin. In Claude Code "
                    "run '/plugin marketplace remove potpie', then delete "
                    f"'{_RETIRED_CLAUDE_PLUGIN_DIR.as_posix()}/'."
                ),
            }
        )
    return removed, leftovers


def _lf_sha256(path: Path) -> str | None:
    """SHA-256 of a file with CRLF folded to LF (text writes on Windows add CR)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _is_retired_claude_plugin(directory: Path) -> bool:
    """Is this the plugin directory Potpie used to install, by its own manifest?"""
    try:
        manifest = json.loads(
            (directory / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("name") == "potpie"


__all__ = [
    "InstallResult",
    "UninstallResult",
    "install_agent_bundle",
    "install_bundle",
    "iter_template_files",
    "prune_empty_dirs",
    "resolve_install_root",
    "uninstall_bundle",
]
