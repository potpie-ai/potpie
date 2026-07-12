"""Host-routed CLI command surface.

One Typer sub-app (or top-level command set) per ``cli-flow.md`` command group,
each routing ``CLI -> PotpieRuntime -> service(s) -> ports``. Assembled into the root
app by ``build_app`` (see ``potpie.cli.main``).
"""

from __future__ import annotations

from potpie.cli.commands import (
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
