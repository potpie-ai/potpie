"""``DefaultSkillManager`` — catalog + per-harness install drift.

Owns the catalog/drift logic; delegates the where/how of installation to a
registered :class:`AgentTargetPort` per harness. Built over the static builtin
catalog and the (POC) Claude target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, cast
from urllib.parse import urlsplit

from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_engine.adapters.outbound.skills.claude_target import (
    ProjectAgentTarget,
)
from potpie_context_engine.adapters.outbound.skills.agent_installer import (
    validate_packaged_skill_command_snippets,
)
from potpie_context_engine.adapters.outbound.skills.bundle_catalog import (
    RECOMMENDED_SKILL_IDS,
    catalog_by_id,
)
from potpie_context_core.ports.agent_context import SkillNudge
from potpie_context_engine.domain.ports.services.skill_manager import (
    AgentTargetPort,
    SkillInfo,
    SkillOperationResult,
    SkillStatus,
)

#: URL schemes a catalog add could plausibly fetch a skill from. Anything else
#: is a typo wearing a colon.
_FETCHABLE_SCHEMES = frozenset({"http", "https", "git", "ssh"})

#: What a skill source has to contain to be one.
SKILL_MANIFEST = "SKILL.md"


def validate_skill_source(source: str) -> None:
    """Refuse a source no catalog add could ever resolve.

    ``skills add`` took anything: a path with a typo in it, a bare hostname, an
    ``ftp://`` URL, the empty string. Every one of them came back at exit 0, so
    the only way to discover the skill had not been registered was that it never
    showed up in ``skills list`` — a check nobody runs after a command that said
    it worked.
    """
    text = (source or "").strip()
    if not text:
        raise ValueError(
            f"Pass a skill source: a directory containing a {SKILL_MANIFEST}, "
            f"or an https URL."
        )
    scheme = urlsplit(text).scheme.lower()
    if scheme:
        if scheme not in _FETCHABLE_SCHEMES:
            raise ValueError(
                f"Cannot fetch a skill over '{scheme}'. "
                f"Supported: {', '.join(sorted(_FETCHABLE_SCHEMES))}."
            )
        return
    candidate = Path(text).expanduser()
    if not candidate.exists():
        raise ValueError(
            f"No such skill source: {candidate}. A local source is a directory "
            f"containing a {SKILL_MANIFEST}; a remote one needs its scheme "
            f"(e.g. 'https://…')."
        )
    if candidate.is_dir():
        if not (candidate / SKILL_MANIFEST).is_file():
            raise ValueError(f"{candidate} is not a skill: it has no {SKILL_MANIFEST}.")
        return
    if candidate.name != SKILL_MANIFEST:
        raise ValueError(
            f"{candidate} is not a skill: point at a directory containing a "
            f"{SKILL_MANIFEST}, or at the {SKILL_MANIFEST} itself."
        )


@dataclass(slots=True)
class DefaultSkillManager:
    targets: dict[str, AgentTargetPort] = field(default_factory=dict)

    def _target(self, agent: str) -> AgentTargetPort:
        target = self.targets.get(agent)
        if target is None:
            raise ValueError(
                f"No install target registered for agent '{agent}'. "
                f"Known: {', '.join(sorted(self.targets)) or '(none)'}."
            )
        return target

    def _target_for_scope(
        self, *, agent: str, scope: str = "global", path: str | None = None
    ) -> AgentTargetPort:
        normalized_scope = scope.strip().lower() if scope else "global"
        if normalized_scope == "global":
            return self._target(agent)
        if normalized_scope == "project":
            # The same refusal global scope makes, for the same reason. A
            # project target was built for *any* string, and its unknown harness
            # fell through to the default ``.agents/skills`` layout — so
            # ``skills list --agent clawd --scope project`` answered a typo with
            # a complete, entirely plausible listing of eleven skills, all
            # ``installed: false``, for a harness that does not exist.
            self._target(agent)
            return ProjectAgentTarget(agent=agent, path=Path(path or "."))
        raise ValueError("scope must be 'global' or 'project'")

    @staticmethod
    def _metadata(
        target: AgentTargetPort,
        *,
        scope: str,
        support_files: tuple[str, ...] = (),
        unavailable: tuple[str, ...] = (),
        not_installed: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        # ``target_root`` first: it is where files actually land. Reading
        # ``path`` off a project target reported the ``--path`` the caller typed,
        # which is a different directory from the repo root the installer
        # resolves to whenever they point at a subdirectory.
        root = (
            getattr(target, "target_root", None)
            or getattr(target, "skills_root", None)
            or getattr(target, "path", None)
        )
        metadata: dict[str, Any] = {"scope": scope}
        if root is not None:
            metadata["target_root"] = str(root)
        if support_files:
            metadata["support_files"] = list(support_files)
        if unavailable:
            metadata["unavailable"] = list(unavailable)
        if not_installed:
            metadata["not_installed"] = list(not_installed)
        return metadata

    @staticmethod
    def _available(target: AgentTargetPort) -> frozenset[str] | None:
        """Ids this harness's bundle carries, or ``None`` if it will not say.

        The catalog is built from one bundle and installs go to several, so
        "in the catalog" and "installable for this harness" are different
        questions. Conflating them made the Claude plugin — which ships ten of
        the eleven skills — report the eleventh in ``changed`` on every single
        run, because nothing was written and nothing could be read back.
        """
        lister = getattr(target, "available", None)
        if not callable(lister):
            return None
        return frozenset(cast("Iterable[str]", lister()))

    def _agents_carrying(self, sid: str, *, besides: str) -> list[str]:
        """Registered harnesses whose bundle really does carry this skill.

        The way out used to be spelled ``--agent claude`` whatever harness was
        asked and whatever skill was named, so the one caller who most needs it
        — the one already on ``claude`` — was answered with the command that had
        just been refused. A target that will not say what it carries is taken
        at its word, the same way :meth:`_matches_bundle` takes it.
        """
        return [
            name
            for name, target in sorted(self.targets.items())
            if name != besides
            and ((available := self._available(target)) is None or sid in available)
        ]

    def _unavailable_skill_error(self, sid: str, agent: str) -> ValueError:
        alternatives = self._agents_carrying(sid, besides=agent)
        way_out = (
            f"Install it for a harness that carries it, e.g. 'potpie skills "
            f"install {sid} --agent {alternatives[0]}'."
            if alternatives
            else "No harness bundle in this build carries it."
        )
        return ValueError(
            f"Skill '{sid}' is in the catalog but this build's {agent} bundle "
            f"does not carry it, so there is nothing to install. {way_out}"
        )

    def _missing_skill_error(
        self,
        sid: str,
        catalog: dict[str, SkillInfo],
        *,
        agent: str,
        installed: bool = False,
    ) -> ValueError:
        """Which refusal a *named* id has earned: unknown, or not carried here.

        Availability is asked first, because a harness bundle is what can
        actually install — but that put an id which is in *neither* the bundle
        nor the catalog on the bundle's refusal, which opens by asserting the
        skill "is in the catalog" and closes by naming a harness to install it
        on. Both halves are false for an id that does not exist, and the second
        sends the caller after a skill no harness has. Only the real targets
        answer ``available()``, so every test that drove a fake one saw the
        right message while the product printed the wrong one.
        """
        if sid not in catalog:
            return self._unknown_skill_error(sid, catalog, installed=installed)
        return self._unavailable_skill_error(sid, agent)

    @staticmethod
    def _matches_bundle(
        target: AgentTargetPort, skill_id: str, *, path: str | None
    ) -> bool:
        """Is the installed skill's content still what the bundle carries?

        A target that cannot answer is taken at its word rather than assumed
        drifted — the alternative reinstalls every skill on every command for
        anyone running a target this build does not know about.
        """
        checker = getattr(target, "matches_bundle", None)
        if not callable(checker):
            return True
        return bool(checker(skill_id=skill_id, path=path))

    def _is_current(
        self,
        target: AgentTargetPort,
        sid: str,
        *,
        installed: Mapping[str, str],
        version: str,
        path: str | None,
    ) -> bool:
        """Both halves of "already installed": the right version, intact on disk."""
        if installed.get(sid) != version:
            return False
        return self._matches_bundle(target, sid, path=path)

    @staticmethod
    def _unknown_skill_error(
        sid: str, catalog: dict[str, SkillInfo], *, installed: bool = False
    ) -> ValueError:
        """The refusal owed to a caller who typed an id by hand.

        Every op here iterates either a set the caller named or a set the
        product chose. Skipping a *named* id reports the operation as done for
        a skill that does not exist, so the next session runs without the
        context the harness was told it had. An id that is installed but gone
        from the bundle gets the only next step that works — there is no
        version to install over it, so the honest instruction is to remove it.
        """
        if installed:
            return ValueError(
                f"Skill '{sid}' is installed but this build's bundle no longer "
                f"carries it, so there is no version to update to. "
                f"Remove it with 'potpie skills remove {sid}'."
            )
        return ValueError(
            f"Unknown skill '{sid}'. "
            f"Available: {', '.join(sorted(catalog)) or '(none)'}."
        )

    @staticmethod
    def _install_support_files(
        target: AgentTargetPort, *, path: str | None = None
    ) -> tuple[str, ...]:
        """Write the harness's instruction file / slash commands; say which.

        Returns the files it touched so the caller can report them. They used to
        be written on every install, including one that named a single skill,
        and never appeared in the result at all — so a command asked for one
        skill and silently edited the user's ``CLAUDE.md``.
        """
        installer = getattr(target, "install_support_files", None)
        if not callable(installer):
            return ()
        result = installer(path=path)
        created = tuple(getattr(result, "created", ()) or ())
        updated = tuple(getattr(result, "updated", ()) or ())
        return created + updated

    @staticmethod
    def _remove_support_files(
        target: AgentTargetPort, *, path: str | None = None
    ) -> tuple[str, ...]:
        """Take the harness's instruction file / slash commands back out.

        Symmetric with :meth:`_install_support_files`, and only reached by the
        sweep for the same reason: these files belong to the bundle as a whole,
        not to any one id. Without it ``skills remove --all`` deleted every
        skill directory and left ``CLAUDE.md`` and the ``/potpie-*`` slash
        commands in place, so the harness went on advertising commands whose
        skills were gone.
        """
        remover = getattr(target, "remove_support_files", None)
        if not callable(remover):
            return ()
        result = remover(path=path)
        return tuple(getattr(result, "removed", ()) or ())

    def list(
        self, *, agent: str = "claude", scope: str = "global", path: str | None = None
    ) -> list[SkillInfo]:
        catalog = catalog_by_id()
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        installed = target.installed()
        out: list[SkillInfo] = []
        for sid, info in catalog.items():
            ver = installed.get(sid)
            # The *installed* version, not the catalog's. Reporting the catalog
            # version beside every installed skill made this listing
            # structurally unable to show drift: it gave a clean bill of health
            # to the same skills ``skills status`` was calling outdated.
            drifted = ver is not None and not self._matches_bundle(
                target, sid, path=path
            )
            out.append(
                SkillInfo(
                    id=info.id,
                    title=info.title,
                    version=info.version,
                    description=info.description,
                    installed=ver is not None,
                    installed_version=ver,
                    drifted=drifted,
                )
            )
        return out

    def install(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        catalog = catalog_by_id()
        requested = bool(skill_id)
        ids = [skill_id] if requested else list(RECOMMENDED_SKILL_IDS)
        changed: list[str] = []
        unavailable: list[str] = []
        available = self._available(target)
        installed = target.installed()
        for sid in ids:
            if available is not None and sid not in available:
                # Named, it is a refusal: there is no version of this skill for
                # this harness and reporting it installed is how a caller comes
                # to believe an agent has context it does not. Swept, it is
                # reported instead of skipped — a bundle that cannot carry
                # everything should say so, not quietly install ten of eleven.
                if requested:
                    raise self._missing_skill_error(str(sid), catalog, agent=agent)
                unavailable.append(str(sid))
                continue
            info = catalog.get(sid)
            if info is None:
                # A *recommended* id the catalog does not carry is a packaging
                # gap to walk past — the rest of the bundle still installs. An
                # id the caller typed is a typo, and skipping it reported the
                # install as done for a skill that does not exist, so the next
                # command runs without the context it was told it had.
                if requested:
                    raise self._unknown_skill_error(sid, catalog)
                continue
            if self._is_current(
                target, sid, installed=installed, version=info.version, path=path
            ):
                continue
            validate_packaged_skill_command_snippets(skill_ids=(sid,))
            target.install(skill_id=sid, version=info.version, path=path)
            changed.append(sid)
        # Only the sweep gets to touch the harness's own files. Naming one skill
        # id is a request for that skill, and honouring it by also rewriting
        # CLAUDE.md — a file the user wrote — is a bigger edit than the one they
        # asked for, made without saying so.
        support = () if requested else self._install_support_files(target, path=path)
        return SkillOperationResult(
            agent=agent,
            operation="install",
            changed=tuple(changed),
            metadata=self._metadata(
                target,
                scope=scope,
                support_files=support,
                unavailable=tuple(unavailable),
            ),
        )

    def update(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        all_: bool = False,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        if all_ and skill_id:
            # The refusal ``remove`` already makes, for the same reason: the two
            # name different sets, so honouring one silently discards the other
            # — and the discarded half is the id the caller typed. ``--all``
            # used to win outright, which made ``skills update <id> --all``
            # report a sweep of everything installed as though it had been the
            # answer to a question about one skill.
            raise ValueError("pass either a skill id or --all, not both")
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        catalog = catalog_by_id()
        installed = target.installed()
        # No id at all is the sweep, so ``--all`` is only ever the explicit
        # spelling of the default; what it must not be is a way to make a named
        # id disappear.
        requested = bool(skill_id)
        ids = [skill_id] if requested else list(installed)
        changed: list[str] = []
        available = self._available(target)
        for sid in ids:
            if available is not None and sid not in available:
                # Same split as ``install``: the sweep only ever walks ids that
                # are already installed, so reaching this at all means a named
                # id, and there is no version of it for this harness to move to
                # — unless the catalog has never heard of it either, which is a
                # typo and gets told so rather than sent to another harness.
                raise self._missing_skill_error(
                    str(sid), catalog, agent=agent, installed=sid in installed
                )
            info = catalog.get(sid)
            if info is None:
                # Sweeping what is installed must walk past an entry this
                # build's bundle no longer carries. A named id is different:
                # skipping it exited 0 with `changed: []`, which reads exactly
                # like "already up to date" for a skill the caller never had.
                if requested:
                    raise self._unknown_skill_error(
                        sid, catalog, installed=sid in installed
                    )
                continue
            if self._is_current(
                target, sid, installed=installed, version=info.version, path=path
            ):
                continue
            validate_packaged_skill_command_snippets(skill_ids=(sid,))
            # ``path=`` was dropped here while ``install`` passed it, so the two
            # commands could resolve the same ``--path`` to different roots.
            target.install(skill_id=sid, version=info.version, path=path)
            changed.append(sid)
        support = () if requested else self._install_support_files(target, path=path)
        return SkillOperationResult(
            agent=agent,
            operation="update",
            changed=tuple(changed),
            metadata=self._metadata(target, scope=scope, support_files=support),
        )

    def remove(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        all_: bool = False,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        if all_ and skill_id:
            raise ValueError("pass either a skill id or --all, not both")
        if not all_ and not skill_id:
            raise ValueError("pass a skill id or --all")
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        catalog = catalog_by_id()
        installed = target.installed()
        ids = list(installed) if all_ else [skill_id]
        changed: list[str] = []
        not_installed: list[str] = []
        for sid in ids:
            if sid is None:
                continue
            if sid not in installed:
                # "Not installed" is a real answer here — the end state the
                # caller asked for already holds, and `removed: []` says so, so
                # a second `skills remove` must not fail. An id that neither the
                # catalog nor the harness has ever heard of is a typo instead,
                # and answering a typo with `removed: []` is how a caller comes
                # to believe a skill is gone from a harness it was never in.
                if not all_ and sid not in catalog:
                    raise self._unknown_skill_error(sid, catalog)
                # Reported rather than merely skipped: `removed: []` alone is
                # the same answer this command gives when it removed the last
                # skill a moment ago, so the caller could not tell "already
                # gone" from "gone from a harness you did not mean".
                not_installed.append(sid)
                continue
            target.remove(skill_id=sid)
            changed.append(sid)
        # Only the sweep owns the harness's own files — the mirror of the rule
        # `install` follows. It runs whether or not any skill was removed:
        # support files outlive the skill directories (a hand-deleted
        # `.claude/skills` leaves them orphaned), so gating on `changed` would
        # make the second `remove --all` the one that finally cannot clean up.
        support = self._remove_support_files(target, path=path) if all_ else ()
        return SkillOperationResult(
            agent=agent,
            operation="remove",
            changed=tuple(changed),
            metadata=self._metadata(
                target,
                scope=scope,
                support_files=support,
                not_installed=tuple(not_installed),
            ),
        )

    def status(
        self, *, agent: str, path: str | None = None, scope: str = "global"
    ) -> SkillStatus:
        catalog = catalog_by_id()
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        installed = target.installed()
        available = self._available(target)
        installed_infos: list[SkillInfo] = []
        missing: list[SkillInfo] = []
        outdated: list[SkillInfo] = []
        for sid in RECOMMENDED_SKILL_IDS:
            # "Recommended" has to mean recommended *for this harness*. A skill
            # its bundle cannot carry is not missing, it is inapplicable, and
            # listing it as missing produced a permanent nag whose install
            # command has nothing to install.
            if available is not None and sid not in available:
                continue
            info = catalog[sid]
            ver = installed.get(sid)
            if ver is None:
                missing.append(info)
                continue
            # Content drift lands in ``outdated`` beside a stale version, and
            # for the same reason: both mean "what the harness is loading is not
            # what this build ships", and both are fixed by the same reinstall.
            # A file corrupted at its recorded version was previously reported
            # as healthy by every command in this group.
            drifted = not self._matches_bundle(target, sid, path=path)
            record = SkillInfo(
                id=info.id,
                title=info.title,
                version=info.version,
                description=info.description,
                installed=True,
                installed_version=ver,
                drifted=drifted,
            )
            if ver != info.version or drifted:
                outdated.append(record)
            else:
                installed_infos.append(record)
        return SkillStatus(
            agent=agent,
            installed=tuple(installed_infos),
            missing=tuple(missing),
            outdated=tuple(outdated),
        )

    def nudge(self, *, agent: str) -> SkillNudge:
        try:
            st = self.status(agent=agent)
        except ValueError:
            # Unknown harness → empty nudge rather than an error in status().
            return SkillNudge(agent=agent)
        missing = tuple(s.id for s in st.missing)
        outdated = tuple(s.id for s in st.outdated)
        cmd = None
        if missing or outdated:
            cmd = f"potpie skills install --agent {agent}"
        return SkillNudge(
            agent=agent, missing=missing, outdated=outdated, install_command=cmd
        )

    def add(self, *, source: str) -> SkillOperationResult:
        """Register an external skill into the catalog — not built yet.

        Two refusals rather than one line of prose at exit 0. A source that
        could never work is a typo and gets said so by name; a source that
        *would* work gets the documented not-implemented contract, because
        "catalog add not implemented (source=…)" printed as a success — exit 0,
        no error envelope — is how ``skills add ./typo`` and ``skills add
        ftp://anything`` both came back looking like they had registered
        something.
        """
        validate_skill_source(source)
        raise CapabilityNotImplemented(
            "skills.catalog.add",
            detail=(
                f"'{source}' looks like a usable skill source, but this build "
                f"cannot register external skills into the catalog yet."
            ),
            recommended_next_action=(
                "install a packaged skill instead with 'potpie skills install "
                "[<id>]', or list what this build carries with 'potpie skills list'"
            ),
        )


__all__ = ["DefaultSkillManager", "validate_skill_source"]
