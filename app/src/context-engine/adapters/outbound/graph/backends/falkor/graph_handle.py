"""Build / hold a single FalkorDB graph handle (lite or server mode).

Both ``FalkorDBGraphWriter`` and ``FalkorDBClaimQueryStore`` need to talk
to the *same* underlying instance — for FalkorDBLite that's load-bearing,
because two handles on one db file would each spawn a redis-server.
``FalkorGraphBackend`` builds one provider and threads it into both
adapters; this module owns the construction.
"""

from __future__ import annotations

import os
from typing import Any

from domain.ports.settings import ContextEngineSettingsPort


def build_falkordb_graph(settings: ContextEngineSettingsPort, *, mode: str) -> Any:
    """Build a FalkorDB graph handle from settings.

    ``mode="lite"`` → embedded FalkorDBLite via ``redislite``, backed by a
    local file: no server, no Docker. ``mode="server"`` → connect to a
    running FalkorDB over a redis URL; needs the optional ``falkordb`` client.
    Both expose the same ``graph.query(...)`` → ``result.header`` /
    ``result.result_set`` surface the writer + reader rely on.
    """
    name = settings.falkordb_graph_name()
    if mode == "server":
        url = settings.falkordb_url()
        if not url:
            raise RuntimeError(
                "falkor server mode requires a URL — set FALKORDB_URL "
                "(or CONTEXT_ENGINE_FALKORDB_URL), or use the falkor_lite profile"
            )
        from falkordb import FalkorDB

        return FalkorDB.from_url(url).select_graph(name)
    if mode != "lite":
        raise ValueError(f"unknown falkor mode '{mode}' (expected 'lite' or 'server')")
    # Lite (default for the falkor_lite profile): embedded FalkorDBLite over a
    # local file — no server.
    from redislite.falkordb_client import FalkorDB as LiteFalkorDB

    path = settings.falkordb_lite_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return LiteFalkorDB(path).select_graph(name)


class FalkorDBGraphProvider:
    """Lazily build and memoize **one** shared FalkorDB graph handle.

    Lazy so ``build_backend("falkor_lite")`` never connects at wiring time —
    the embedded redis-server only starts when something actually issues a
    query.
    """

    def __init__(self, settings: ContextEngineSettingsPort, *, mode: str) -> None:
        self._settings = settings
        self._mode = mode
        self._graph: Any | None = None

    def __call__(self) -> Any:
        if self._graph is None:
            self._graph = build_falkordb_graph(self._settings, mode=self._mode)
        return self._graph


__all__ = ["FalkorDBGraphProvider", "build_falkordb_graph"]
