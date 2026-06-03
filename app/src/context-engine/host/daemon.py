"""``Daemon`` — local host lifecycle (thin skeleton).

The daemon shell is the local background process for lifecycle, auth, IPC,
health, and logs — it is *not* the business layer. In this skeleton the host
runs in-process, so the daemon reports a synthetic "in-process" liveness and the
real process-management commands are TODO.

    TODO(stage-N): real daemon — install/start/stop as a background process,
    Unix-socket IPC, log files under the pot home, version/health endpoint.

Liveness and readiness are separate (see observability.md): the daemon can be
live while a backend or semantic index is not ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.outbound.pots.local_pot_store import default_home
from domain.errors import CapabilityNotImplemented
from domain.lifecycle import SKIPPED, SetupPlan, StepResult


@dataclass(slots=True)
class Daemon:
    """In-process daemon stand-in with a real lifecycle interface."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True

    def status(self) -> dict[str, Any]:
        return {
            "up": True,
            "mode": "in_process" if self.in_process else "detached",
            "home": str(self.home),
            "detail": "in-process host; no detached daemon yet (TODO)",
        }

    def health(self) -> dict[str, Any]:
        # Liveness only — readiness is the services'/backend's concern.
        return {"live": True, "mode": "in_process"}

    def logs(self, *, follow: bool = False) -> list[str]:
        log_file = self.home / "daemon.log"
        if not log_file.exists():
            return []
        return log_file.read_text(encoding="utf-8").splitlines()

    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        """Idempotent setup seam: make sure the host is running.

        In-process hosts have nothing to start, so this is a no-op that reports
        ``skipped``. A detached daemon owner fills this with install+start (idempotent).
        """
        if self.in_process:
            return StepResult(
                step="daemon.ensure",
                state=SKIPPED,
                detail="in-process host; no detached daemon to start",
                metadata={"mode": "in_process"},
            )
        raise CapabilityNotImplemented(
            "host.daemon.ensure",
            detail="detached daemon install/start not implemented",
            recommended_next_action="implement detached daemon lifecycle",
        )

    def install(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "host.daemon.install",
            detail="detached daemon install not implemented",
            recommended_next_action="host runs in-process; no install needed for the POC",
        )

    def start(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "host.daemon.start",
            recommended_next_action="host runs in-process; no start needed for the POC",
        )

    def stop(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "host.daemon.stop",
            recommended_next_action="host runs in-process; nothing to stop",
        )

    def restart(self) -> dict[str, Any]:
        raise CapabilityNotImplemented(
            "host.daemon.restart",
            recommended_next_action="host runs in-process; nothing to restart",
        )


__all__ = ["Daemon"]
