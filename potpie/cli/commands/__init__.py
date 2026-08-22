"""Typed-client and Potpie-service CLI command surface.

One Typer sub-app (or top-level command set) per ``cli-flow.md`` command group,
routing context operations through ``EngineClient`` and root-owned operations
through finite Potpie services. Assembled into the root app by ``build_app``
(see ``potpie/cli/main.py``).
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
