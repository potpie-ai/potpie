"""The local host shell — the in-process facade over the three services.

``HostShell`` (built by ``potpie.services.host_wiring.build_host_shell``) is the one
object every inbound adapter binds to. ``Daemon`` is the local lifecycle shell.
"""

from __future__ import annotations

from potpie.daemon.daemon import Daemon
from potpie.daemon.shell import HostShell, LedgerFacade

__all__ = ["Daemon", "HostShell", "LedgerFacade"]
