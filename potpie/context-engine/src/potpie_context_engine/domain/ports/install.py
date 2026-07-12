"""``Installer`` — the packaging/OS lifecycle seam.

Makes the ``potpie`` CLI available on PATH and registers the daemon as an OS
service (systemd/launchd). Folded into ``potpie setup`` as a hard step that is
**skipped when already present** (idempotent), so re-running setup is safe. This
is the seam the installation-flow owner fills; it is an outbound adapter, not a
service, because its work is environment-specific (pip/pipx/homebrew/deb).
"""

from __future__ import annotations

from typing import Protocol

from potpie_context_engine.domain.lifecycle import StepResult


class Installer(Protocol):
    """CLI-on-PATH + OS service-unit registration."""

    def is_installed(self) -> bool:
        """True when the CLI is reachable and the service unit is registered."""
        ...

    def install_cli(self) -> StepResult:
        """Put ``potpie`` on PATH for the current platform."""
        ...

    def register_service(self) -> StepResult:
        """Register the daemon as an OS service (systemd/launchd)."""
        ...

    def uninstall(self) -> StepResult:
        """Reverse install_cli + register_service."""
        ...


__all__ = ["Installer"]
