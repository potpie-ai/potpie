"""Context-engine local daemon lifecycle fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.outbound.pots.local_pot_store import default_home
from domain.errors import CapabilityNotImplemented
from domain.lifecycle import SKIPPED, SetupPlan, StepResult


@dataclass(slots=True)
class InProcessDaemonLifecycle:
    """Library-owned lifecycle that never starts a detached daemon."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True

    def discovery(self) -> dict[str, str] | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "up": True,
            "mode": "in_process",
            "home": str(self.home),
            "detail": "in-process host; detached daemon is owned by root potpie",
        }

    def health(self) -> dict[str, Any]:
        return {"live": True, "mode": "in_process"}

    def logs(self, *, follow: bool = False) -> list[str]:
        return []

    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        return StepResult(
            "daemon.ensure",
            SKIPPED,
            "in-process host; no detached daemon to start",
            metadata={"mode": "in_process"},
        )

    def install(self) -> dict[str, Any]:
        return {
            "installed": False,
            "detail": "detached daemon installation is owned by root potpie",
        }

    def start(self, *, backend: str | None = None) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "daemon.start",
            detail="potpie-context-engine does not own the detached daemon",
            recommended_next_action="run this command through the root 'potpie' CLI",
        )

    def stop(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "daemon.stop",
            detail="potpie-context-engine does not own the detached daemon",
            recommended_next_action="run this command through the root 'potpie' CLI",
        )

    def restart(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "daemon.restart",
            detail="potpie-context-engine does not own the detached daemon",
            recommended_next_action="run this command through the root 'potpie' CLI",
        )


__all__ = ["InProcessDaemonLifecycle"]
