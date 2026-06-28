"""The local host shell — the in-process facade over the three services.

``HostShell`` (built by ``bootstrap.host_wiring.build_host_shell``) is the one
object every inbound adapter binds to. ``Daemon`` is the local lifecycle shell.
"""

from __future__ import annotations

from host.shell import HostShell, LedgerFacade

__all__ = ["HostShell", "LedgerFacade"]
