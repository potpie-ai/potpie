"""Install packaged AGENTS.md / CLAUDE.md and skills into agent targets."""

from __future__ import annotations

import importlib
import inspect
import re
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from potpie_context_engine.adapters.outbound.skills.template_resources import (
    NO_TEMPLATE_RESOURCES,
    TemplateResourceProvider,
)
from potpie_context_engine.domain.errors import CapabilityNotImplemented

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


def _template_files_root(template_resources: TemplateResourceProvider):
    return template_resources.files_root()


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


def _iter_bundle_files(
    bundle_name: str,
    *,
    template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES,
) -> list[tuple[Path, str]]:
    """Return packaged template files from the named bundle as (repo-relative path, UTF-8 text)."""
    root = _template_files_root(template_resources).joinpath("templates", bundle_name)
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


def iter_template_files(
    *, template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES
) -> list[tuple[Path, str]]:
    """Return agent_bundle template files (default / codex path)."""
    return _iter_bundle_files("agent_bundle", template_resources=template_resources)


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
    *,
    skill_ids: Iterable[str] | None = None,
    template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES,
) -> None:
    """Validate packaged Potpie CLI snippets before install/update.

    This intentionally validates only ``potpie`` commands in bash fences. Skills
    may contain other shell commands whose correctness depends on the user's repo.
    """
    selected = _normalize_skill_ids(skill_ids)
    for bundle_name in ("agent_bundle", "claude_plugin"):
        for rel_path, content in _iter_bundle_files(
            bundle_name, template_resources=template_resources
        ):
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
    specs: dict[tuple[str, ...], frozenset[str]] = {}
    _collect_typer_command_specs(_potpie_cli_app(), path=(), out=specs)
    return specs


@lru_cache(maxsize=1)
def _potpie_root_options() -> frozenset[str]:
    app = _potpie_cli_app()
    callback = app.registered_callback
    if callback is None or callback.callback is None:
        return frozenset()
    return _callback_option_decls(callback.callback)


def _potpie_cli_app():
    try:
        module = importlib.import_module("potpie" + ".cli.main")
    except ModuleNotFoundError as exc:
        raise CapabilityNotImplemented(
            "skills.cli_command_specs",
            detail="Potpie CLI command metadata is unavailable in this runtime.",
            recommended_next_action=(
                "run skill installation through the root 'potpie' CLI package"
            ),
        ) from exc
    return module.app


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
    template_resources: TemplateResourceProvider,
    force: bool,
    include: Callable[[Path], bool] | None = None,
    remap: Callable[[Path], Path | None] | None = None,
    merge_files: frozenset[str] = _DEFAULT_MERGE_FILES,
) -> None:
    for rel_path, content in _iter_bundle_files(
        bundle_name, template_resources=template_resources
    ):
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
    template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES,
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
        template_resources=template_resources,
        force=force,
        include=lambda rel: _include_selected_skills(rel, selected),
        remap=lambda rel: Path(rel.as_posix()[len(_SOURCE_SKILLS_PREFIX) :]),
    )
    return result


def install_global_agent_instructions(
    root: str | Path,
    *,
    agent: str = "default",
    template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES,
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
        template_resources=template_resources,
        force=force,
        include=lambda rel: rel.as_posix() == filename,
        merge_files=frozenset({filename}),
    )
    return result


def install_agent_bundle(
    path: str | Path = ".",
    *,
    agent: str = "default",
    template_resources: TemplateResourceProvider = NO_TEMPLATE_RESOURCES,
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
        _install_bundle(
            root,
            "claude_bundle",
            result,
            template_resources=template_resources,
            force=force,
        )
        _install_bundle(
            root,
            "agent_bundle",
            result,
            template_resources=template_resources,
            force=force,
            include=lambda rel: _include_selected_skills(rel, selected),
            remap=_claude_skills_bundle_remap,
        )
    elif normalized == "claude-plugin":
        _install_bundle(
            root,
            "claude_plugin",
            result,
            template_resources=template_resources,
            force=force,
            remap=_claude_plugin_remap,
        )
    elif normalized == "cursor":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            template_resources=template_resources,
            force=force,
            include=lambda rel: (
                rel.as_posix() == "AGENTS.md" or _include_selected_skills(rel, selected)
            ),
            remap=_cursor_bundle_remap,
        )
    elif normalized == "opencode":
        _install_bundle(
            root,
            "agent_bundle",
            result,
            template_resources=template_resources,
            force=force,
            include=lambda rel: _include_selected_skills(rel, selected),
            remap=_opencode_bundle_remap,
        )
    else:
        _install_bundle(
            root,
            "agent_bundle",
            result,
            template_resources=template_resources,
            force=force,
            include=lambda rel: (
                rel.as_posix() == "AGENTS.md" or _include_selected_skills(rel, selected)
            ),
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
