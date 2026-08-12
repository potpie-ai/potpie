"""Agent harness install targets for Potpie's packaged skills."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.adapters.outbound.skills.agent_installer import (
    InstallResult,
    available_skill_ids,
    install_global_agent_instructions,
    install_agent_bundle,
    install_skill_bundle,
    project_skill_path,
    resolve_install_root,
)
from potpie_context_engine.adapters.outbound.skills.bundle_catalog import (
    RECOMMENDED_SKILL_IDS,
)
from potpie_context_engine.adapters.outbound.skills.harness_home import harness_home

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _project_manifest_slug(root: Path) -> str:
    """A per-project manifest suffix: a readable name plus a collision-proof digest.

    The version manifest used to be keyed by agent and scope alone, so every
    repository on the machine shared one record of "which skills are installed
    at project scope". Installing in one project reported the others up to date;
    ``skills remove --all`` in one marked every other project on the machine
    fully outdated. The digest is what actually separates them — the name is
    there so a human can tell which file belongs to which checkout.
    """
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    name = _SLUG_RE.sub("-", root.name).strip("-") or "project"
    return f"{name}_{digest}"


@dataclass(slots=True)
class FileBackedAgentTarget:
    """Install packaged Potpie skills into one harness-specific skills root."""

    agent: str
    skills_root: Path
    instructions_root: Path | None = None
    instructions_agent: str | None = None
    home: Path = field(default_factory=default_home)
    scope: str = "global"

    @property
    def target_root(self) -> Path:
        return self.skills_root.expanduser()

    @property
    def _path(self) -> Path:
        return self.home / f"skills_{self.agent}_{self.scope}.json"

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _save(self, data: Mapping[str, str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(dict(data), fh, indent=2)

    def _skill_file(self, skill_id: str) -> Path:
        return self.skills_root.expanduser() / skill_id / "SKILL.md"

    def installed(self) -> Mapping[str, str]:
        manifest = self._load()
        installed: dict[str, str] = {}
        for sid in RECOMMENDED_SKILL_IDS:
            if self._skill_file(sid).exists():
                installed[sid] = manifest.get(sid, "unknown")
        return installed

    def available(self) -> frozenset[str]:
        return available_skill_ids(agent=self.agent)

    def matches_bundle(self, *, skill_id: str, path: str | None = None) -> bool:
        """Is what is on disk byte-identical to what :meth:`install` would write?

        A version integer cannot answer this. A ``SKILL.md`` truncated by a
        failed write, or hand-edited, kept its recorded version, so ``install``
        and ``update --all`` both exited 0 with ``changed: []`` and the harness
        went on loading a broken skill with no way to repair it short of
        deleting the directory by hand.
        """
        root = Path(path).expanduser() if path else self.skills_root
        result = install_skill_bundle(
            root, skill_ids=(skill_id,), force=True, dry_run=True
        )
        return not (result.created or result.updated)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        root = Path(path).expanduser() if path else self.skills_root
        install_skill_bundle(root, skill_ids=(skill_id,), force=True)
        data = self._load()
        if (root.expanduser() / skill_id / "SKILL.md").exists():
            data[skill_id] = version
        self._save(data)

    def install_support_files(self, *, path: str | None = None) -> InstallResult | None:
        del path
        if self.instructions_root is None:
            return None
        return install_global_agent_instructions(
            self.instructions_root,
            agent=self.instructions_agent or self.agent,
            force=True,
        )

    def remove(self, *, skill_id: str) -> None:
        shutil.rmtree(self.skills_root.expanduser() / skill_id, ignore_errors=True)
        data = self._load()
        data.pop(skill_id, None)
        self._save(data)


@dataclass(slots=True)
class ProjectAgentTarget:
    """Install packaged Potpie skills into a repository-local harness path."""

    agent: str = "claude"
    path: Path = Path(".")
    home: Path = field(default_factory=default_home)
    scope: str = "project"

    @property
    def target_root(self) -> Path:
        """Where files actually land — the repo root, not the path passed in.

        ``install`` has always resolved to the nearest git root, so reporting
        the raw ``--path`` as ``metadata.target_root`` named a directory nothing
        was written to whenever the caller pointed at a subdirectory.
        """
        return resolve_install_root(self.path)

    @property
    def _path(self) -> Path:
        return (
            self.home
            / f"skills_{self.agent}_{self.scope}_{_project_manifest_slug(self.target_root)}.json"
        )

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _save(self, data: Mapping[str, str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(dict(data), fh, indent=2)

    def installed(self) -> Mapping[str, str]:
        manifest = self._load()
        installed: dict[str, str] = {}
        for sid in RECOMMENDED_SKILL_IDS:
            if project_skill_path(self.path, agent=self.agent, skill_id=sid).exists():
                installed[sid] = manifest.get(sid, "unknown")
        return installed

    def available(self) -> frozenset[str]:
        return available_skill_ids(agent=self.agent)

    def matches_bundle(self, *, skill_id: str, path: str | None = None) -> bool:
        """See :meth:`FileBackedAgentTarget.matches_bundle`."""
        root = Path(path) if path else self.path
        result = install_agent_bundle(
            root,
            agent=self.agent,
            skill_ids=(skill_id,),
            force=True,
            support_files=False,
            dry_run=True,
        )
        return not (result.created or result.updated)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        root = Path(path) if path else self.path
        # Support files are the caller's *other* request; see
        # ``install_support_files``. Bundling them in here is what made
        # ``skills install potpie-cli`` also write CLAUDE.md, two slash commands
        # and a second skill, none of them named in ``changed``.
        install_agent_bundle(
            root,
            agent=self.agent,
            skill_ids=(skill_id,),
            force=True,
            support_files=False,
        )
        data = self._load()
        if project_skill_path(root, agent=self.agent, skill_id=skill_id).exists():
            data[skill_id] = version
        self._save(data)

    def install_support_files(self, *, path: str | None = None) -> InstallResult:
        root = Path(path) if path else self.path
        return install_agent_bundle(
            root, agent=self.agent, skill_ids=(), force=True, support_files=True
        )

    def remove(self, *, skill_id: str) -> None:
        shutil.rmtree(
            project_skill_path(self.path, agent=self.agent, skill_id=skill_id).parent,
            ignore_errors=True,
        )
        data = self._load()
        data.pop(skill_id, None)
        self._save(data)


@dataclass(slots=True)
class ProjectOnlyAgentTarget:
    """A harness that only has a project-scope install, refusing global scope.

    Registered so the agent is *known* rather than merely absent. Left out of
    the registry, ``--agent claude-plugin`` answered every subcommand with "no
    install target registered … Known: claude, codex, cursor, opencode" — a
    listing that implies the harness is unsupported, while its bundle ships in
    the wheel and installs fine one flag away.
    """

    agent: str
    reason: str
    scope: str = "global"

    def _refuse(self) -> None:
        raise ValueError(self.reason)

    def installed(self) -> Mapping[str, str]:
        self._refuse()
        return {}  # pragma: no cover - _refuse always raises

    def matches_bundle(self, *, skill_id: str, path: str | None = None) -> bool:
        del skill_id, path
        self._refuse()
        return False  # pragma: no cover - _refuse always raises

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        del skill_id, version, path
        self._refuse()

    def remove(self, *, skill_id: str) -> None:
        del skill_id
        self._refuse()


class CursorAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs: dict[str, Any] = {"home": home} if home is not None else {}
        super().__init__(
            agent="cursor",
            skills_root=harness_home() / ".cursor" / "skills",
            **kwargs,
        )


class ClaudeAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs: dict[str, Any] = {"home": home} if home is not None else {}
        super().__init__(
            agent="claude",
            skills_root=harness_home() / ".claude" / "skills",
            instructions_root=harness_home() / ".claude",
            instructions_agent="claude",
            **kwargs,
        )


class OpenCodeAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs: dict[str, Any] = {"home": home} if home is not None else {}
        super().__init__(
            agent="opencode",
            skills_root=harness_home() / ".config" / "opencode" / "skills",
            **kwargs,
        )


class CodexAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs: dict[str, Any] = {"home": home} if home is not None else {}
        super().__init__(
            agent="codex",
            skills_root=harness_home() / ".agents" / "skills",
            instructions_root=harness_home() / ".codex",
            instructions_agent="codex",
            **kwargs,
        )


class ClaudePluginAgentTarget(ProjectOnlyAgentTarget):
    def __init__(self) -> None:
        super().__init__(
            agent="claude-plugin",
            reason=(
                "The Claude Code plugin installs into a project, not a home "
                "directory: it has to keep its '.claude-plugin/plugin.json' as "
                "the plugin root. Re-run with '--scope project --path <repo>'."
            ),
        )


__all__ = [
    "ClaudeAgentTarget",
    "ClaudePluginAgentTarget",
    "CodexAgentTarget",
    "CursorAgentTarget",
    "FileBackedAgentTarget",
    "OpenCodeAgentTarget",
    "ProjectAgentTarget",
    "ProjectOnlyAgentTarget",
]
