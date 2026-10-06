"""Local ``Installer`` — packaging/OS seam.

``is_installed`` reports ``True`` only when the ``potpie`` on ``PATH`` serves
every command the installed agent guidance names (``cli_probe``): agents run
that executable, which may be an older install than the one running setup.
When it does, the setup orchestrator **skips** the install step. The install
bodies — putting ``potpie`` on PATH and registering an OS service unit
(systemd/launchd) — are fail-closed stubs for the installation-flow owner to
fill. They are environment-specific (pip/pipx/homebrew/deb), which is why this
is an outbound adapter, not a service.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from potpie.setup.cli_probe import probe_cli_surface
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.lifecycle import StepResult


@dataclass(slots=True)
class LocalInstaller:
    """CLI-on-PATH + OS service-unit registration."""

    def is_installed(self) -> bool:
        executable = shutil.which("potpie")
        if executable is None:
            return False
        return bool(probe_cli_surface(executable)["ok"])

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
