"""``DefaultSkillManager`` — catalog + per-harness install drift.

Owns the catalog/drift logic; delegates the where/how of installation to a
registered :class:`AgentTargetPort` per harness. Built over the static builtin
catalog and the (POC) Claude target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from potpie.context_engine.adapters.outbound.skills.claude_target import ProjectAgentTarget
from potpie.context_engine.adapters.outbound.skills.bundle_catalog import (
    RECOMMENDED_SKILL_IDS,
    catalog_by_id,
)
from potpie.context_engine.domain.ports.agent_context import SkillNudge
from potpie.context_engine.domain.ports.services.skill_manager import (
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
        ids = [skill_id] if skill_id else list(RECOMMENDED_SKILL_IDS)
        changed: list[str] = []
        installed = target.installed()
        for sid in ids:
            info = catalog.get(sid)
            if info is None:
                continue
            if installed.get(sid) == info.version:
                continue
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
        target = self._target_for_scope(agent=agent, scope=scope, path=path)
        catalog = catalog_by_id()
        installed = target.installed()
        ids = list(installed) if (all_ or skill_id is None) else [skill_id]
        changed: list[str] = []
        for sid in ids:
            info = catalog.get(sid)
            if info and installed.get(sid) != info.version:
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
        installed = target.installed()
        ids = list(installed) if all_ else [skill_id]
        changed: list[str] = []
        for sid in ids:
            if sid is None or sid not in installed:
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
