"""The install target for Potpie's packaged skills: one harness at one scope."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping

from potpie.config.local_paths import default_home
from potpie.skills.bundle import bundle_skill_ids
from potpie.skills.catalog import RECOMMENDED_SKILL_IDS
from potpie.skills.harness_home import harness_home
from potpie.skills.harnesses import HarnessLayout, harness_layout
from potpie.skills.installer import (
    InstallResult,
    UninstallResult,
    install_bundle,
    resolve_install_root,
    uninstall_bundle,
)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_SCOPES = ("global", "project")


def _read_version_manifest(path: Path) -> dict[str, str]:
    """The recorded ``skill id -> version`` map, or an empty one it can repair.

    The manifest is a *cache* of what install last wrote; the files on disk are
    the truth about what is installed. So an unreadable one is answered with
    "no recorded versions", which surfaces as ``installed_version="unknown"``,
    lands every present skill in ``skills status --outdated``, and is repaired
    by the reinstall that report already tells the user to run.

    Catching only ``JSONDecodeError`` is not enough: a manifest holding valid
    JSON of the wrong *shape* — a list, a string, anything a stray write leaves
    behind — would raise ``AttributeError`` out of ``skills list`` and be
    reported as an internal error, for a cache file the next install rewrites.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, Mapping):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


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
class AgentTarget:
    """Install packaged Potpie skills for one harness, globally or in one repo.

    Where files land comes from the harness's row in
    :data:`~potpie.skills.harnesses.HARNESS_LAYOUTS`: under the harness home at
    ``global`` scope, under the repository containing ``path`` at ``project``
    scope. What was installed — versions, content hashes, disabled skills — is
    recorded under ``home``.
    """

    agent: str
    scope: str = "global"
    path: Path = Path(".")
    home: Path = field(default_factory=default_home)
    _layout: HarnessLayout = field(init=False, repr=False)
    _harness_home: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError("scope must be 'global' or 'project'")
        self._layout = harness_layout(self.agent)
        self._harness_home = harness_home()

    # --- where files land -----------------------------------------------------

    @property
    def target_root(self) -> Path:
        """Where files actually land: the skills root, or the repo root.

        At project scope this is the repo root, not the path passed in:
        ``install`` has always resolved to the nearest git root, so reporting
        the raw ``--path`` named a directory nothing was written to whenever
        the caller pointed at a subdirectory.
        """
        if self.scope == "global":
            return self.skills_root
        return resolve_install_root(self.path)

    @property
    def skills_root(self) -> Path:
        """The directory holding one subdirectory per installed skill."""
        if self.scope == "global":
            return self._harness_home / self._layout.global_skills
        return self.target_root / self._layout.project_skills

    def _placement(self, path: str | None) -> tuple[Path, PurePosixPath]:
        """``(root, skills directory under it)`` for an operation on ``path``.

        ``path`` overrides the default the way it always has: a skills root at
        global scope, a directory inside the repository at project scope.
        """
        if self.scope == "global":
            root = Path(path).expanduser() if path else self.skills_root
            return root, PurePosixPath(".")
        repo = resolve_install_root(Path(path) if path else self.path)
        return repo, self._layout.project_skills

    def _instructions(self, path: str | None) -> tuple[Path, PurePosixPath] | None:
        """``(root, instruction file under it)``, or ``None`` if the harness has none."""
        if self.scope == "global":
            rel = self._layout.global_instructions
            if rel is None:
                return None
            # Rooted at the file's own directory, so results name `CLAUDE.md`.
            return self._harness_home / rel.parent, PurePosixPath(rel.name)
        rel = self._layout.project_instructions
        if rel is None:
            return None
        return resolve_install_root(Path(path) if path else self.path), rel

    @property
    def _sweeps_retired_claude_files(self) -> bool:
        # Only repositories ever received the retired slash commands and plugin.
        return self.scope == "project" and self._layout.sweeps_retired_claude_files

    def _skill_file(self, skill_id: str, *, path: str | None = None) -> Path:
        root, skills_dir = self._placement(path)
        return root / skills_dir / skill_id / "SKILL.md"

    # --- install records ------------------------------------------------------

    @property
    def _path(self) -> Path:
        name = f"skills_{self.agent}_{self.scope}"
        if self.scope == "project":
            name = f"{name}_{_project_manifest_slug(self.target_root)}"
        return self.home / f"{name}.json"

    @property
    def _hash_path(self) -> Path:
        return self._path.with_name(
            self._path.name.replace("skills_", "skill_hashes_", 1)
        )

    @property
    def _disabled_path(self) -> Path:
        return self._path.with_name(
            self._path.name.replace("skills_", "skill_disabled_", 1)
        )

    def _save_to(self, path: Path, data: Mapping[str, str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dict(data), fh, indent=2)

    # --- AgentTargetPort ------------------------------------------------------

    def installed(self) -> Mapping[str, str]:
        manifest = _read_version_manifest(self._path)
        return {
            sid: manifest.get(sid, "unknown")
            for sid in RECOMMENDED_SKILL_IDS
            if self._skill_file(sid).exists()
        }

    def available(self) -> frozenset[str]:
        return bundle_skill_ids()

    def matches_bundle(self, *, skill_id: str, path: str | None = None) -> bool:
        """Is what is on disk byte-identical to what :meth:`install` would write?

        A version integer cannot answer this. A ``SKILL.md`` truncated by a
        failed write, or hand-edited, kept its recorded version, so ``install``
        and ``update --all`` both exited 0 with ``changed: []`` and the harness
        went on loading a broken skill with no way to repair it short of
        deleting the directory by hand.
        """
        root, skills_dir = self._placement(path)
        result = install_bundle(
            root,
            skills_dir=skills_dir,
            skill_ids=(skill_id,),
            force=True,
            dry_run=True,
        )
        return not (result.created or result.updated)

    def locally_modified(self, *, skill_id: str) -> bool:
        expected = _read_version_manifest(self._hash_path).get(skill_id)
        current = _file_digest(self._skill_file(skill_id))
        return expected is not None and current is not None and current != expected

    def disabled(self) -> frozenset[str]:
        return frozenset(_read_version_manifest(self._disabled_path))

    def set_disabled(self, *, skill_id: str, disabled: bool) -> None:
        state = _read_version_manifest(self._disabled_path)
        if disabled:
            state[skill_id] = "disabled"
        else:
            state.pop(skill_id, None)
        self._save_to(self._disabled_path, state)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        # The instruction file is the caller's *other* request (see
        # ``install_instructions``): bundling it in here is what made
        # ``skills install potpie-cli`` also edit CLAUDE.md without naming it.
        root, skills_dir = self._placement(path)
        install_bundle(root, skills_dir=skills_dir, skill_ids=(skill_id,), force=True)
        skill_md = self._skill_file(skill_id, path=path)
        versions = _read_version_manifest(self._path)
        if skill_md.exists():
            versions[skill_id] = version
        self._save_to(self._path, versions)
        digest = _file_digest(skill_md)
        if digest:
            hashes = _read_version_manifest(self._hash_path)
            hashes[skill_id] = digest
            self._save_to(self._hash_path, hashes)

    def install_instructions(self, *, path: str | None = None) -> InstallResult | None:
        placement = self._instructions(path)
        if placement is None:
            return None
        root, rel = placement
        return install_bundle(
            root,
            skill_ids=(),
            instructions=rel,
            sweep_retired_claude_files=self._sweeps_retired_claude_files,
            force=True,
        )

    def remove_instructions(self, *, path: str | None = None) -> UninstallResult | None:
        """The mirror of :meth:`install_instructions`."""
        placement = self._instructions(path)
        if placement is None:
            return None
        root, rel = placement
        return uninstall_bundle(
            root,
            instructions=rel,
            sweep_retired_claude_files=self._sweeps_retired_claude_files,
        )

    def remove(self, *, skill_id: str) -> None:
        root, skills_dir = self._placement(None)
        uninstall_bundle(root, skills_dir=skills_dir, skill_ids=(skill_id,))
        versions = _read_version_manifest(self._path)
        versions.pop(skill_id, None)
        self._save_to(self._path, versions)


__all__ = ["AgentTarget"]
