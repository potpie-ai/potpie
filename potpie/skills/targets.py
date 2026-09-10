"""Agent harness install targets for Potpie's packaged skills."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from potpie.config.local_paths import default_home, harness_home
from potpie.skills.catalog import (
    RECOMMENDED_SKILL_IDS,
)
from potpie.skills.installer import (
    install_agent_bundle,
    install_global_agent_instructions,
    install_skill_bundle,
    project_skill_path,
)


# Where each harness installs skills when POTPIE_HARNESS_HOME is unset. Used
# only to recognise the default root, so its manifest keeps its historical
# filename and existing installs are not orphaned.
_DEFAULT_SKILLS_SUBPATH: dict[str, tuple[str, ...]] = {
    "claude": (".claude", "skills"),
    "cursor": (".cursor", "skills"),
    "opencode": (".config", "opencode", "skills"),
    "codex": (".agents", "skills"),
}


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
    def _path(self) -> Path:
        # The manifest records which version is installed *in this root*.
        # Since POTPIE_HARNESS_HOME can point two roots at one state home, the
        # root is part of the manifest's identity — otherwise installing into
        # one root rewrites the versions reported for the other. The default
        # root keeps the original filename so existing installs still resolve.
        return self.home / f"skills_{self.agent}_{self.scope}{self._root_suffix()}.json"

    def _root_suffix(self) -> str:
        resolved = self.skills_root.expanduser()
        if resolved == self._default_skills_root():
            return ""
        digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
        return f"_{digest}"

    def _default_skills_root(self) -> Path:
        """Where this harness installs when no override is in play."""

        relative = _DEFAULT_SKILLS_SUBPATH.get(self.agent)
        if relative is None:
            return self.skills_root.expanduser()
        return Path.home().joinpath(*relative)

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

    def unmanaged(self) -> tuple[str, ...]:
        return _unmanaged_skill_ids(self.skills_root)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        root = Path(path).expanduser() if path else self.skills_root
        install_skill_bundle(root, skill_ids=(skill_id,), force=True)
        data = self._load()
        if (root / skill_id / "SKILL.md").exists():
            data[skill_id] = version
        self._save(data)

    def install_support_files(self, *, path: str | None = None) -> None:
        del path
        if self.instructions_root is None:
            return
        install_global_agent_instructions(
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
    def _path(self) -> Path:
        return self.home / f"skills_{self.agent}_{self.scope}.json"

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

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

    def unmanaged(self) -> tuple[str, ...]:
        sample = project_skill_path(self.path, agent=self.agent, skill_id="__probe__")
        return _unmanaged_skill_ids(sample.parent.parent)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        root = Path(path) if path else self.path
        install_agent_bundle(root, agent=self.agent, skill_ids=(skill_id,), force=True)
        data = self._load()
        if project_skill_path(root, agent=self.agent, skill_id=skill_id).exists():
            data[skill_id] = version
        self._save(data)

    def install_support_files(self, *, path: str | None = None) -> None:
        root = Path(path) if path else self.path
        install_agent_bundle(root, agent=self.agent, skill_ids=(), force=True)

    def remove(self, *, skill_id: str) -> None:
        shutil.rmtree(
            project_skill_path(self.path, agent=self.agent, skill_id=skill_id).parent,
            ignore_errors=True,
        )
        data = self._load()
        data.pop(skill_id, None)
        self._save(data)


# Skill directories we own are named `potpie-*`; anything matching that shape
# but absent from this build's catalog is an orphan the harness still loads.
_POTPIE_SKILL_PREFIX = "potpie-"


def _unmanaged_skill_ids(skills_root: Path | None) -> tuple[str, ...]:
    if skills_root is None:
        return ()
    root = Path(skills_root).expanduser()
    try:
        entries = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    except OSError:
        return ()
    known = set(RECOMMENDED_SKILL_IDS)
    return tuple(
        name
        for name in entries
        if name.startswith(_POTPIE_SKILL_PREFIX)
        and name not in known
        and (root / name / "SKILL.md").exists()
    )


class CursorAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs = {"home": home} if home is not None else {}
        super().__init__(
            agent="cursor",
            skills_root=harness_home() / ".cursor" / "skills",
            **kwargs,
        )


class ClaudeAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs = {"home": home} if home is not None else {}
        super().__init__(
            agent="claude",
            skills_root=harness_home() / ".claude" / "skills",
            instructions_root=harness_home() / ".claude",
            instructions_agent="claude",
            **kwargs,
        )


class OpenCodeAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs = {"home": home} if home is not None else {}
        super().__init__(
            agent="opencode",
            skills_root=harness_home() / ".config" / "opencode" / "skills",
            **kwargs,
        )


class CodexAgentTarget(FileBackedAgentTarget):
    def __init__(self, *, home: Path | None = None) -> None:
        kwargs = {"home": home} if home is not None else {}
        super().__init__(
            agent="codex",
            skills_root=harness_home() / ".agents" / "skills",
            instructions_root=harness_home() / ".codex",
            instructions_agent="codex",
            **kwargs,
        )


__all__ = [
    "ClaudeAgentTarget",
    "CodexAgentTarget",
    "CursorAgentTarget",
    "FileBackedAgentTarget",
    "OpenCodeAgentTarget",
    "ProjectAgentTarget",
]
