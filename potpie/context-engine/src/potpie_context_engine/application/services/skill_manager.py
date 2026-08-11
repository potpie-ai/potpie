"""``DefaultSkillManager`` — catalog + per-harness install drift.

Owns the catalog/drift logic; delegates the where/how of installation to a
registered :class:`AgentTargetPort` per harness. Built over the static builtin
catalog and the (POC) Claude target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
            return ProjectAgentTarget(agent=agent, path=Path(path or "."))
        raise ValueError("scope must be 'global' or 'project'")

    @staticmethod
    def _metadata(target: AgentTargetPort, *, scope: str) -> dict[str, str]:
        root = getattr(target, "skills_root", None) or getattr(target, "path", None)
        metadata = {"scope": scope}
        if root is not None:
            metadata["target_root"] = str(root)
        return metadata

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
    ) -> None:
        installer = getattr(target, "install_support_files", None)
        if callable(installer):
            installer(path=path)

    def list(
        self, *, agent: str = "claude", scope: str = "global", path: str | None = None
    ) -> list[SkillInfo]:
        catalog = catalog_by_id()
        installed = self._target_for_scope(
            agent=agent, scope=scope, path=path
        ).installed()
        out: list[SkillInfo] = []
        for sid, info in catalog.items():
            ver = installed.get(sid)
            out.append(
                SkillInfo(
                    id=info.id,
                    title=info.title,
                    version=info.version,
                    description=info.description,
                    installed=ver is not None,
                    installed_version=ver,
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
        installed = target.installed()
        for sid in ids:
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
            if installed.get(sid) == info.version:
                continue
            validate_packaged_skill_command_snippets(skill_ids=(sid,))
            target.install(skill_id=sid, version=info.version, path=path)
            changed.append(sid)
        self._install_support_files(target, path=path)
        return SkillOperationResult(
            agent=agent,
            operation="install",
            changed=tuple(changed),
            metadata=self._metadata(target, scope=scope),
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
        for sid in ids:
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
            if installed.get(sid) == info.version:
                continue
            validate_packaged_skill_command_snippets(skill_ids=(sid,))
            target.install(skill_id=sid, version=info.version)
            changed.append(sid)
        self._install_support_files(target, path=path)
        return SkillOperationResult(
            agent=agent,
            operation="update",
            changed=tuple(changed),
            metadata=self._metadata(target, scope=scope),
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
                continue
            target.remove(skill_id=sid)
            changed.append(sid)
        return SkillOperationResult(
            agent=agent,
            operation="remove",
            changed=tuple(changed),
            metadata=self._metadata(target, scope=scope),
        )

    def status(
        self, *, agent: str, path: str | None = None, scope: str = "global"
    ) -> SkillStatus:
        catalog = catalog_by_id()
        installed = self._target_for_scope(
            agent=agent, scope=scope, path=path
        ).installed()
        installed_infos: list[SkillInfo] = []
        missing: list[SkillInfo] = []
        outdated: list[SkillInfo] = []
        for sid in RECOMMENDED_SKILL_IDS:
            info = catalog[sid]
            ver = installed.get(sid)
            if ver is None:
                missing.append(info)
            elif ver != info.version:
                outdated.append(info)
            else:
                installed_infos.append(
                    SkillInfo(
                        id=info.id,
                        title=info.title,
                        version=info.version,
                        description=info.description,
                        installed=True,
                        installed_version=ver,
                    )
                )
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
        # TODO(stage-N): register a local-path/URL skill into the catalog.
        return SkillOperationResult(
            agent="(catalog)",
            operation="add",
            detail=f"catalog add not implemented (source={source})",
        )


__all__ = ["DefaultSkillManager"]
