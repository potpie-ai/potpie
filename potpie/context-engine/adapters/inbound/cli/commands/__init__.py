"""Host-routed CLI command surface.

One Typer sub-app (or top-level command set) per ``cli-flow.md`` command group,
each routing ``CLI -> HostShell -> service(s) -> ports``. Assembled into the root
app by ``build_app`` (see ``adapters/inbound/cli/host_cli.py``).
"""

from __future__ import annotations

from adapters.inbound.cli.commands import (
    auth,
    bootstrap,
    cloud,
    daemon,
    graph,
    ledger,
    pots,
    query,
    skills,
)

__all__ = [
    "auth",
    "bootstrap",
    "cloud",
    "daemon",
    "graph",
    "ledger",
    "pots",
    "query",
    "skills",
]
