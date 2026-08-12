"""Install packaged AGENTS.md / CLAUDE.md and skills into agent targets."""

from __future__ import annotations

import inspect
import re
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

_MANAGED_MARKER_RE = re.compile(
    r"<!-- (?:context-engine|potpie)-start -->.*?<!-- (?:context-engine|potpie)-end -->",
    re.DOTALL,
)
_DEFAULT_MERGE_FILES = frozenset({"AGENTS.md", "CLAUDE.md"})
_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

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


@dataclass
class UninstallResult:
    """What a support-file sweep took back out — the mirror of ``InstallResult``.

    Its own shape rather than a reused ``InstallResult``: "created/updated" have
    no meaning for a removal, and a caller that reported one as the other would
    tell the user a file was written when it was deleted.
    """

    root: str
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ok"] = True
        return data


def _unwritable_target_error(
    target: Path, exc: OSError, *, verb: str = "write"
) -> ValueError:
    """The refusal owed to a caller whose install target cannot be written.

    A read-only repository (or a ``--path`` under someone else's ownership)
    surfaced as a bare ``PermissionError``. In-process that is an "Unexpected
    internal error"; across the daemon it is an unclassified 500, which the CLI
    reports as ``unavailable`` at exit 2 with "check backend/daemon readiness
    with 'potpie doctor'" — so the operator diagnoses a healthy daemon while the
    actual repair is one ``chmod`` on a directory the message never named.
    """
    blocked = _nearest_existing_dir(target)
    error = ValueError(
        f"Cannot {verb} {target}: {exc.strerror or exc}. "
        f"The harness directory is not writable, so nothing was changed there."
    )
    error.recommended_next_action = (  # type: ignore[attr-defined]
        f"make '{blocked}' writable, or point somewhere else with '--path <dir>'"
    )
    return error


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


def resolve_install_root(path: str | Path) -> Path:
    """Prefer the nearest git repo root; otherwise install into the given path."""
    target = Path(path).resolve()
    if target.is_file():
        raise ValueError(f"Expected a directory path, got file: {target}")
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return target


@lru_cache(maxsize=8)
def _iter_bundle_files(bundle_name: str) -> tuple[tuple[Path, str], ...]:
    """Return packaged template files from the named bundle as (repo-relative path, UTF-8 text).

    Cached because it is now on a read path, not just a write one: every
    content-drift check re-walks the bundle it is comparing against, and
    ``skills status`` runs one per recommended skill. The bundle ships inside the
    installed wheel and cannot change under a running process; tests that edit
    templates in place call :func:`clear_bundle_file_cache`.
    """
    root = resources.files("potpie.cli").joinpath("templates", bundle_name)
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
    return tuple(sorted(out, key=lambda item: item[0].as_posix()))


def clear_bundle_file_cache() -> None:
    """Test helper: drop cached reads of the packaged template bundles."""
    _iter_bundle_files.cache_clear()


def iter_template_files() -> tuple[tuple[Path, str], ...]:
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


def _is_skill_markdown(rel_path: Path) -> bool:
    return rel_path.name == "SKILL.md" and "skills" in rel_path.parts


def _selected_skill_matches(rel_path: Path, selected: frozenset[str] | None) -> bool:
    if selected is None:
        return True
    sid = _skill_id_for_path(rel_path) or _skill_id_from_generic_skill_path(rel_path)
    return sid in selected


def _skill_id_from_generic_skill_path(rel_path: Path) -> str | None:
    parts = rel_path.parts
    for idx, part in enumerate(parts[:-1]):
        if part == "skills" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _include_selected_skills(rel_path: Path, selected: frozenset[str] | None) -> bool:
    sid = _skill_id_for_path(rel_path)
    return sid is not None and (selected is None or sid in selected)


def validate_packaged_skill_command_snippets(
    *, skill_ids: Iterable[str] | None = None
) -> None:
    """Validate packaged Potpie CLI snippets before install/update.

    This intentionally validates only ``potpie`` commands in bash fences. Skills
    may contain other shell commands whose correctness depends on the user's repo.
    """
    selected = _normalize_skill_ids(skill_ids)
    for bundle_name in ("agent_bundle", "claude_plugin"):
        for rel_path, content in _iter_bundle_files(bundle_name):
            if not _is_skill_markdown(rel_path):
                continue
            if not _selected_skill_matches(rel_path, selected):
                continue
            validate_skill_command_snippets(content, rel_path=rel_path)


def validate_skill_command_snippets(content: str, *, rel_path: Path) -> None:
    errors: list[str] = []
    for line in _iter_potpie_bash_lines(content):
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as exc:
            errors.append(f"{line!r}: {exc}")
            continue
        if not tokens or tokens[0] != "potpie":
            continue
        error = _validate_potpie_command_tokens(tokens)
        if error:
            errors.append(error)
    if errors:
        prefix = f"invalid Potpie command snippets in {rel_path.as_posix()}: "
        raise ValueError(prefix + "; ".join(errors))


def _iter_potpie_bash_lines(content: str) -> Iterable[str]:
    for block in _BASH_BLOCK_RE.findall(content):
        pending = ""
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].lstrip()
            if pending:
                line = f"{pending} {line}"
            if line.endswith("\\"):
                pending = line[:-1].rstrip()
                continue
            pending = ""
            if line == "potpie" or line.startswith("potpie "):
                yield line
        if pending and (pending == "potpie" or pending.startswith("potpie ")):
            yield pending


def _validate_potpie_command_tokens(tokens: list[str]) -> str | None:
    specs = _potpie_command_option_specs()
    root_options = _potpie_root_options()
    idx = 1
    while idx < len(tokens) and tokens[idx].startswith("-"):
        opt = _option_name(tokens[idx])
        if opt not in root_options:
            return f"{' '.join(tokens)} uses unsupported root option {opt}"
        idx += 1

    match: tuple[tuple[str, ...], int, frozenset[str]] | None = None
    for end in range(idx + 1, len(tokens) + 1):
        token = tokens[end - 1]
        if token.startswith("-"):
            break
        path = tuple(tokens[idx:end])
        options = specs.get(path)
        if options is not None:
            match = (path, end, options)
    if match is None:
        command = " ".join(tokens[idx : idx + 3]) or "(missing command)"
        return f"{' '.join(tokens)} uses unknown potpie command {command!r}"

    path, end, command_options = match
    for token in tokens[end:]:
        if not token.startswith("-") or token == "-":
            continue
        opt = _option_name(token)
        if opt not in command_options:
            command = " ".join(path)
            return (
                f"{' '.join(tokens)} uses unsupported option {opt} for potpie {command}"
            )
    return None


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


@lru_cache(maxsize=1)
def _potpie_command_option_specs() -> dict[tuple[str, ...], frozenset[str]]:
    from potpie.cli.main import app

    specs: dict[tuple[str, ...], frozenset[str]] = {}
    _collect_typer_command_specs(app, path=(), out=specs)
    return specs


@lru_cache(maxsize=1)
def _potpie_root_options() -> frozenset[str]:
    from potpie.cli.main import app

    callback = app.registered_callback
    if callback is None or callback.callback is None:
        return frozenset()
    return _callback_option_decls(callback.callback)


def _collect_typer_command_specs(
    typer_app, *, path: tuple[str, ...], out: dict[tuple[str, ...], frozenset[str]]
) -> None:
    for command in typer_app.registered_commands:
        if command.callback is None:
            continue
        out[(*path, _typer_command_name(command))] = _callback_option_decls(
            command.callback
        )
    for group in typer_app.registered_groups:
        _collect_typer_command_specs(
            group.typer_instance,
            path=(*path, group.name),
            out=out,
        )


def _typer_command_name(command) -> str:
    if command.name:
        return str(command.name)
    return command.callback.__name__.replace("_", "-")


def _callback_option_decls(callback: Callable[..., object]) -> frozenset[str]:
    from typer.models import OptionInfo

    options: set[str] = set()
    for parameter in inspect.signature(callback).parameters.values():
        default = parameter.default
        if not isinstance(default, OptionInfo):
            continue
        for decl in default.param_decls:
            options.update(_split_option_decl(str(decl)))
    return frozenset(options)


def _split_option_decl(decl: str) -> tuple[str, ...]:
    if not decl.startswith("-"):
        return ()
    out: list[str] = []
    for part in decl.split("/"):
        if part.startswith("-"):
            out.append(part)
    return tuple(out)


def _install_file(
    install_root: Path,
    rel_path: Path,
    content: str,
    result: InstallResult,
    *,
    force: bool,
    dry_run: bool = False,
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
        if not dry_run:
            _write_installed_file(target, content)
        result.updated.append(rel_path.as_posix())
        return
    if not dry_run:
        _write_installed_file(target, content)
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
    dry_run: bool = False,
) -> None:
    for rel_path, content in _iter_bundle_files(bundle_name):
        if include is not None and not include(rel_path):
            continue
        if _is_skill_markdown(rel_path):
            validate_skill_command_snippets(content, rel_path=rel_path)
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
            if not dry_run:
                _write_installed_file(target, merged)
            if action == "created":
                result.created.append(out_path.as_posix())
            else:
                result.updated.append(out_path.as_posix())
            continue

        _install_file(
            install_root, out_path, content, result, force=force, dry_run=dry_run
        )


def _uninstall_bundle(
    install_root: Path,
    bundle_name: str,
    result: UninstallResult,
    *,
    include: Callable[[Path], bool] | None = None,
    remap: Callable[[Path], Path | None] | None = None,
    merge_files: frozenset[str] = _DEFAULT_MERGE_FILES,
    dry_run: bool = False,
) -> None:
    """Take back exactly the files :func:`_install_bundle` would have written.

    Same bundle, same ``include``, same ``remap`` — because a removal built from
    its own list of paths is a second opinion about what install owns, and the
    two drift the moment a file is added to a bundle. That drift is the defect
    itself: ``skills remove --all`` deleted every skill directory and left the
    harness instruction file and the ``/potpie-*`` slash commands loaded.
    """
    for rel_path, _content in _iter_bundle_files(bundle_name):
        if include is not None and not include(rel_path):
            continue
        out_path = rel_path if remap is None else remap(rel_path)
        if out_path is None:
            continue
        _uninstall_file(
            install_root,
            out_path,
            result,
            merge=out_path.name in merge_files,
            dry_run=dry_run,
        )


def _uninstall_file(
    install_root: Path,
    rel_path: Path,
    result: UninstallResult,
    *,
    merge: bool,
    dry_run: bool = False,
) -> None:
    target = install_root / rel_path
    if not target.exists():
        result.unchanged.append(rel_path.as_posix())
        return
    if merge:
        # A file the user also writes in: strip Potpie's managed block and keep
        # whatever else is there. Deleting a hand-written CLAUDE.md because
        # Potpie once appended to it would be a far bigger removal than the one
        # the caller asked for.
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
        return
    if not dry_run:
        _remove_installed_file(target)
        prune_empty_dirs(target.parent, stop_at=install_root)
    result.removed.append(rel_path.as_posix())


def _remove_installed_file(target: Path) -> None:
    try:
        target.unlink()
    except OSError as exc:
        raise _unwritable_target_error(target, exc, verb="remove") from exc


def prune_empty_dirs(directory: Path, *, stop_at: Path) -> None:
    """Drop directories a removal just emptied, up to but excluding ``stop_at``.

    Stops at the first non-empty parent, so a ``.claude/`` that still holds
    skills — or anything the user put there — survives. Without it a full
    uninstall left the shape of the install behind: an empty
    ``.claude/potpie-plugin/skills/`` that reads, to anyone who opens the repo,
    as a plugin that is still there.
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
    dry_run: bool = False,
) -> InstallResult:
    """Install selected packaged skills directly into a skills root.

    ``skills_root`` is the directory that contains one subdirectory per skill,
    for example ``~/.cursor/skills`` or ``~/.agents/skills``.

    ``dry_run`` classifies every file exactly as a real run would — created /
    updated / unchanged / skipped — and writes nothing. That is what makes
    content drift detectable: the comparison is the installer's own, so it
    cannot drift from what install actually does the way a second, parallel
    "is it current?" implementation would.
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
        dry_run=dry_run,
    )
    return result


def install_global_agent_instructions(
    root: str | Path,
    *,
    agent: str = "default",
    force: bool = True,
    dry_run: bool = False,
) -> InstallResult:
    """Install compact global instructions for harnesses with file-based rules.

    The project bundle is intentionally detailed. This global bundle stays tiny
    because it can be loaded into every prompt across repositories.
    """
    install_root = Path(root).expanduser().resolve()
    result = InstallResult(root=str(install_root))
    filename = _global_instructions_filename(agent)
    if filename is None:
        return result

    _install_bundle(
        install_root,
        "global_agent_bundle",
        result,
        force=force,
        include=lambda rel: rel.as_posix() == filename,
        merge_files=frozenset({filename}),
        dry_run=dry_run,
    )
    return result


def uninstall_global_agent_instructions(
    root: str | Path,
    *,
    agent: str = "default",
    dry_run: bool = False,
) -> UninstallResult:
    """Strip the managed section from a harness's global instruction file."""
    install_root = Path(root).expanduser().resolve()
    result = UninstallResult(root=str(install_root))
    filename = _global_instructions_filename(agent)
    if filename is None:
        return result

    _uninstall_bundle(
        install_root,
        "global_agent_bundle",
        result,
        include=lambda rel: rel.as_posix() == filename,
        merge_files=frozenset({filename}),
        dry_run=dry_run,
    )
    return result


def _global_instructions_filename(agent: str) -> str | None:
    """Which global instruction file a harness reads, or ``None`` if it has none."""
    normalized = agent.strip().lower() if agent else "default"
    if normalized == "claude":
        return "CLAUDE.md"
    if normalized in {"default", "codex"}:
        return "AGENTS.md"
    return None


def _support_file_include(
    rel_path: Path, selected: frozenset[str] | None, *, support_files: bool
) -> bool:
    """Include a bundle file that is *not* one of the packaged skills.

    Split out because "install the skill I named" and "install this harness's
    supporting files" are two different requests that shared one code path:
    ``skills install potpie-cli`` also wrote ``CLAUDE.md``, two slash commands
    and an entire second skill nobody asked for, and reported only the one id in
    ``changed``.
    """
    del rel_path, selected
    return support_files


def _claude_bundle_include(
    rel_path: Path, selected: frozenset[str] | None, *, support_files: bool
) -> bool:
    """Claude's project bundle carries both kinds of file, so it is split here.

    ``.claude/skills/<id>/`` is a skill and obeys the selection (its
    ``potpie-graph`` copy is byte-identical to the ``agent_bundle`` one that the
    remap below installs). ``CLAUDE.md`` and ``.claude/commands/`` are support
    files.
    """
    sid = _skill_id_from_generic_skill_path(rel_path)
    if sid is not None:
        return selected is None or sid in selected
    return support_files


def _claude_plugin_include(
    rel_path: Path, selected: frozenset[str] | None, *, support_files: bool
) -> bool:
    """Same split for the plugin bundle, whose skills sit at ``skills/<id>/``.

    Everything else — ``.claude-plugin/``, ``commands/``, ``hooks/``, the README
    — is what makes the directory a loadable plugin, so it travels with any
    install that is allowed to write support files.
    """
    sid = _skill_id_from_generic_skill_path(rel_path)
    if sid is not None:
        return selected is None or sid in selected
    return support_files


@dataclass(frozen=True)
class _BundlePlan:
    """One packaged bundle, which of its files apply, and where they land."""

    bundle: str
    include: Callable[[Path], bool]
    remap: Callable[[Path], Path | None] | None = None


def _agent_bundle_plans(
    *, agent: str, selected: frozenset[str] | None, support_files: bool
) -> tuple[_BundlePlan, ...]:
    """The bundles a harness installs from — read by install *and* uninstall.

    One table, because the removal side has to own exactly the set the install
    side wrote. Spelled out twice they drift on the next bundle file added, and
    what drifts away is always the same thing: a support file nothing removes.
    """
    normalized = agent.strip().lower() if agent else "default"
    if normalized not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type {agent!r}. Choose one of: {', '.join(AGENT_TYPES)}"
        )
    if normalized == "claude":
        return (
            _BundlePlan(
                "claude_bundle",
                lambda rel: _claude_bundle_include(
                    rel, selected, support_files=support_files
                ),
            ),
            _BundlePlan(
                "agent_bundle",
                lambda rel: _include_selected_skills(rel, selected),
                _claude_skills_bundle_remap,
            ),
        )
    if normalized == "claude-plugin":
        return (
            _BundlePlan(
                "claude_plugin",
                lambda rel: _claude_plugin_include(
                    rel, selected, support_files=support_files
                ),
                _claude_plugin_remap,
            ),
        )
    if normalized == "cursor":
        return (
            _BundlePlan(
                "agent_bundle",
                lambda rel: (
                    _support_file_include(rel, selected, support_files=support_files)
                    if rel.as_posix() == "AGENTS.md"
                    else _include_selected_skills(rel, selected)
                ),
                _cursor_bundle_remap,
            ),
        )
    if normalized == "opencode":
        return (
            _BundlePlan(
                "agent_bundle",
                lambda rel: _include_selected_skills(rel, selected),
                _opencode_bundle_remap,
            ),
        )
    return (
        _BundlePlan(
            "agent_bundle",
            lambda rel: (
                _support_file_include(rel, selected, support_files=support_files)
                if rel.as_posix() == "AGENTS.md"
                else _include_selected_skills(rel, selected)
            ),
        ),
    )


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    force: bool = False,
    skill_ids: Iterable[str] | None = None,
    support_files: bool = True,
    dry_run: bool = False,
) -> InstallResult:
    """Install agent bundle files into the nearest git repo root under *path*.

    - ``default`` / ``codex``: ``AGENTS.md`` + ``.agents/skills/``
    - ``claude``: ``CLAUDE.md`` (+ ``.claude/`` when present in bundle)
    - ``claude-plugin``: the Claude Code plugin under ``.claude/potpie-plugin/``
    - ``cursor``: ``AGENTS.md`` + ``.cursor/skills/``
    - ``opencode``: ``.opencode/skills/``

    ``support_files=False`` installs only the selected skills, leaving the
    harness's instruction file and slash commands alone — what a caller naming
    one skill id actually asked for.
    """
    root = resolve_install_root(path)
    result = InstallResult(root=str(root))
    selected = _normalize_skill_ids(skill_ids)

    for plan in _agent_bundle_plans(
        agent=agent, selected=selected, support_files=support_files
    ):
        _install_bundle(
            root,
            plan.bundle,
            result,
            force=force,
            include=plan.include,
            remap=plan.remap,
            dry_run=dry_run,
        )

    return result


def uninstall_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    dry_run: bool = False,
) -> UninstallResult:
    """Remove a harness's *support* files — the mirror of ``support_files=True``.

    Skill directories are the target's own business (it removes them one id at a
    time). What this owns is everything the install wrote that no ``changed``
    entry ever named: the instruction file's managed section, the ``/potpie-*``
    slash commands, and — for the Claude Code plugin — the manifest and hooks
    that make the directory loadable. Leaving those behind is why a harness kept
    advertising Potpie slash commands after ``skills remove --all`` had removed
    every skill they refer to.
    """
    root = resolve_install_root(path)
    result = UninstallResult(root=str(root))
    for plan in _agent_bundle_plans(
        agent=agent, selected=frozenset(), support_files=True
    ):
        _uninstall_bundle(
            root,
            plan.bundle,
            result,
            include=plan.include,
            remap=plan.remap,
            dry_run=dry_run,
        )
    return result


def available_skill_ids(*, agent: str = "default") -> frozenset[str]:
    """Skill ids the packaged bundle can actually install for this harness.

    The catalog is built from ``agent_bundle`` alone, and not every harness
    bundle carries every id — the Claude Code plugin ships ten of the eleven.
    Without this the manager reported the missing one in ``changed`` on every
    run: the install wrote nothing, ``installed()`` never saw the file, and the
    next command "installed" it again, forever.
    """
    normalized = agent.strip().lower() if agent else "default"
    if normalized == "claude-plugin":
        bundles = ("claude_plugin",)
    elif normalized == "claude":
        bundles = ("agent_bundle", "claude_bundle")
    else:
        bundles = ("agent_bundle",)
    ids: set[str] = set()
    for bundle_name in bundles:
        for rel_path, _ in _iter_bundle_files(bundle_name):
            if rel_path.name != "SKILL.md":
                continue
            sid = _skill_id_for_path(rel_path) or _skill_id_from_generic_skill_path(
                rel_path
            )
            if sid:
                ids.add(sid)
    return frozenset(ids)


def project_skill_path(root: str | Path, *, agent: str, skill_id: str) -> Path:
    """Return the project-scope SKILL.md path for a harness and skill id."""
    install_root = resolve_install_root(root)
    normalized = agent.strip().lower() if agent else "default"
    if normalized == "cursor":
        return install_root / ".cursor" / "skills" / skill_id / "SKILL.md"
    if normalized == "claude":
        return install_root / ".claude" / "skills" / skill_id / "SKILL.md"
    if normalized == "claude-plugin":
        return install_root / _CLAUDE_PLUGIN_PREFIX / "skills" / skill_id / "SKILL.md"
    if normalized == "opencode":
        return install_root / ".opencode" / "skills" / skill_id / "SKILL.md"
    return install_root / ".agents" / "skills" / skill_id / "SKILL.md"
