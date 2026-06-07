"""``DefaultSkillManager`` — catalog + per-harness install drift.

Owns the catalog/drift logic; delegates the where/how of installation to a
registered :class:`AgentTargetPort` per harness. Built over the static builtin
catalog and the (POC) Claude target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.outbound.skills.builtin_catalog import (
    RECOMMENDED_SKILL_IDS,
    catalog_by_id,
)
from domain.ports.agent_context import SkillNudge
from domain.ports.services.skill_manager import (
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

    def list(self) -> list[SkillInfo]:
        catalog = catalog_by_id()
        # Mark installed-state from the first registered target, if any.
        installed: dict[str, str] = {}
        for target in self.targets.values():
            installed.update(target.installed())
            break
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
        self, *, agent: str, skill_id: str | None = None, path: str | None = None
    ) -> SkillOperationResult:
        target = self._target(agent)
        catalog = catalog_by_id()
        ids = [skill_id] if skill_id else list(RECOMMENDED_SKILL_IDS)
        changed: list[str] = []
        for sid in ids:
            info = catalog.get(sid)
            if info is None:
                continue
            target.install(skill_id=sid, version=info.version, path=path)
            changed.append(sid)
        return SkillOperationResult(
            agent=agent, operation="install", changed=tuple(changed)
        )

    def update(
        self, *, agent: str, skill_id: str | None = None, all_: bool = False
    ) -> SkillOperationResult:
        target = self._target(agent)
        catalog = catalog_by_id()
        installed = target.installed()
        ids = list(installed) if (all_ or skill_id is None) else [skill_id]
        changed: list[str] = []
        for sid in ids:
            info = catalog.get(sid)
            if info and installed.get(sid) != info.version:
                target.install(skill_id=sid, version=info.version)
                changed.append(sid)
        return SkillOperationResult(
            agent=agent, operation="update", changed=tuple(changed)
        )

    def remove(self, *, agent: str, skill_id: str) -> SkillOperationResult:
        self._target(agent).remove(skill_id=skill_id)
        return SkillOperationResult(
            agent=agent, operation="remove", changed=(skill_id,)
        )

    def status(self, *, agent: str) -> SkillStatus:
        catalog = catalog_by_id()
        installed = self._target(agent).installed()
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
