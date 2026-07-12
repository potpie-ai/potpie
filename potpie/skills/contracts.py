"""``SkillManager`` service port.

Skills are CLI-managed recipes that teach an agent harness how to use the
``potpie`` CLI and the four context tools. They are **not** graph facts and
**not** new agent tools. Agents only ever see an advisory ``skills`` block
(``SkillNudge``) inside ``context_status`` listing missing/outdated skills and
the exact install command — installation is a human/CLI action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

# SkillNudge is the advisory block embedded in context_status; it is defined
# in the agent contract module to keep the ports layer acyclic and re-exported
# here so skill code can import it from its own service module.


@dataclass(frozen=True, slots=True)
class SkillNudge:
    """Advisory skill drift rendered by the root product status layer."""

    agent: str
    missing: tuple[str, ...] = ()
    outdated: tuple[str, ...] = ()
    install_command: str | None = None


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """One skill in the catalog or installed for an agent."""

    id: str
    title: str
    version: str
    description: str = ""
    installed: bool = False
    installed_version: str | None = None


@dataclass(frozen=True, slots=True)
class SkillStatus:
    """Per-agent installed-vs-recommended drift for ``skills status``."""

    agent: str
    installed: tuple[SkillInfo, ...] = ()
    missing: tuple[SkillInfo, ...] = ()
    outdated: tuple[SkillInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillOperationResult:
    """Outcome of an install/update/remove/add operation."""

    agent: str
    operation: str
    changed: tuple[str, ...] = ()
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentTargetPort(Protocol):
    """Harness-specific install target (e.g. Claude ``.claude/`` bundle).

    The extension seam for supporting a new agent harness: implement this port
    and register it with the ``SkillManager``. The manager owns catalog/drift
    logic; the target owns where/how files land for one harness.
    """

    @property
    def agent(self) -> str: ...

    def installed(self) -> Mapping[str, str]:
        """Installed skill id -> version for this harness."""
        ...

    def install(
        self, *, skill_id: str, version: str, path: str | None = None
    ) -> None: ...

    def remove(self, *, skill_id: str) -> None: ...


class SkillManager(Protocol):
    """CLI-managed skill catalog and per-agent installation layer."""

    def list(
        self, *, agent: str = "claude", scope: str = "global", path: str | None = None
    ) -> list[SkillInfo]:
        """All catalog skills (with installed/version state when known)."""
        ...

    def install(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        """Install one skill (or the full recommended set when ``skill_id`` is
        ``None``) for an agent harness."""
        ...

    def update(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        all_: bool = False,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        """Update an installed skill, or all of them when ``all_``."""
        ...

    def remove(
        self,
        *,
        agent: str,
        skill_id: str | None = None,
        all_: bool = False,
        path: str | None = None,
        scope: str = "global",
    ) -> SkillOperationResult:
        """Remove one installed skill, or all installed skills when ``all_``."""
        ...

    def status(
        self, *, agent: str, path: str | None = None, scope: str = "global"
    ) -> SkillStatus:
        """Installed-vs-recommended drift for one agent."""
        ...

    def nudge(self, *, agent: str) -> SkillNudge:
        """The advisory block ``context_status`` embeds for an agent."""
        ...

    def add(self, *, source: str) -> SkillOperationResult:
        """Register a local-path or URL skill into the catalog."""
        ...


__all__ = [
    "AgentTargetPort",
    "SkillInfo",
    "SkillManager",
    "SkillNudge",
    "SkillOperationResult",
    "SkillStatus",
]
