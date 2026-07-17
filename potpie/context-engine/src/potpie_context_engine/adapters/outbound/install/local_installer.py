"""POC ``Installer`` — local packaging/OS seam.

In the POC the CLI is already importable / on PATH (you are running it), so
``is_installed`` reports ``True`` and the setup orchestrator **skips** the
install step. The real install bodies — putting ``potpie`` on PATH and
registering an OS service unit (systemd/launchd) — are fail-closed stubs for the
installation-flow owner to fill. They are environment-specific
(pip/pipx/homebrew/deb), which is why this is an outbound adapter, not a service.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_core.domain.errors import CapabilityNotImplemented
from potpie_context_core.domain.lifecycle import StepResult


@dataclass(slots=True)
class LocalInstaller:
    """CLI-on-PATH + OS service-unit registration (POC: reports installed)."""

    def is_installed(self) -> bool:
        # The POC CLI is invoked from an installed/importable entrypoint; the
        # real check inspects PATH + the registered service unit.
        return True

    def install_cli(self) -> StepResult:
        raise CapabilityNotImplemented(
            "host.installer.install_cli",
            detail="putting the potpie CLI on PATH is not implemented",
            recommended_next_action="install via 'pip install potpie' for now",
        )

    def register_service(self) -> StepResult:
        raise CapabilityNotImplemented(
            "host.installer.register_service",
            detail="OS service-unit registration (systemd/launchd) not implemented",
            recommended_next_action="run the host in-process; detached service registration is TODO",
        )

    def uninstall(self) -> StepResult:
        raise CapabilityNotImplemented(
            "host.installer.uninstall",
            detail="uninstall not implemented",
        )


__all__ = ["LocalInstaller"]
