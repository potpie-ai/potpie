"""The install target for Potpie's packaged skills: one harness at one scope."""

from __future__ import annotations

import hashlib
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
from potpie.skills.manifest import Record, SkillManifest

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_SCOPES = ("global", "project")


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _project_manifest_slug(root: Path) -> str:
    """A per-project manifest suffix: a readable name plus a collision-proof digest.

    The manifest used to be keyed by agent and scope alone, so every
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
    recorded in one :class:`~potpie.skills.manifest.SkillManifest` under
    ``home``.
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
    def manifest(self) -> SkillManifest:
        stem = f"{self.agent}_{self.scope}"
        if self.scope == "project":
            stem = f"{stem}_{_project_manifest_slug(self.target_root)}"
        return SkillManifest(self.home, stem)

    # --- AgentTargetPort ------------------------------------------------------

    def installed(self) -> Mapping[str, str]:
        records = self.manifest.read()
        return {
            sid: records.get(sid, {}).get("version", "unknown")
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
        expected = self.manifest.read().get(skill_id, {}).get("sha256")
        current = _file_digest(self._skill_file(skill_id))
        return expected is not None and current is not None and current != expected

    def disabled(self) -> frozenset[str]:
        return frozenset(
            sid
            for sid, record in self.manifest.read().items()
            if record.get("disabled")
        )

    def set_disabled(self, *, skill_id: str, disabled: bool) -> None:
        def change(records: dict[str, Record]) -> None:
            record = records.setdefault(skill_id, {})
            if disabled:
                record["disabled"] = True
            else:
                record.pop("disabled", None)

        self.manifest.update(change)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        # The instruction file is the caller's *other* request (see
        # ``install_instructions``): bundling it in here is what made
        # ``skills install potpie-cli`` also edit CLAUDE.md without naming it.
        root, skills_dir = self._placement(path)
        install_bundle(root, skills_dir=skills_dir, skill_ids=(skill_id,), force=True)
        skill_md = self._skill_file(skill_id, path=path)
        digest = _file_digest(skill_md)

        def change(records: dict[str, Record]) -> None:
            record = records.setdefault(skill_id, {})
            if skill_md.exists():
                record["version"] = version
            if digest:
                record["sha256"] = digest

        self.manifest.update(change)

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

        def change(records: dict[str, Record]) -> None:
            # The hash and the disabled flag outlive the files; only the
            # version said "installed".
            records.get(skill_id, {}).pop("version", None)

        self.manifest.update(change)


__all__ = ["AgentTarget"]
