"""Assemble the daemon's graph-explorer web app.

One factory so the daemon and the tests build the same surface: the loopback
``Host`` guard in front of everything, the daemon token on ``app.state`` for
both halves of the browser handoff, the authenticated ``/ui/api`` router, and
the SPA shell. FastAPI's own schema and docs pages are left off -- they would
be the only unauthenticated JSON on this port.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from potpie.daemon.http.ui.auth import LoopbackHostGuard, UiAuth
from potpie.daemon.http.ui.commits import EngineClientFactory
from potpie.daemon.http.ui.router import build_ui_api_router
from potpie.daemon.http.ui.static import mount_ui_static


def build_ui_app(
    *,
    pots: Any,
    graph: Any,
    backend: Any,
    bearer_token: str,
    engine_client: EngineClientFactory | None = None,
) -> FastAPI:
    """Build the ``/ui`` app over explicit services and this boot's daemon token."""
    app = FastAPI(
        title="potpie-daemon-ui",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.ui_auth = UiAuth(token=bearer_token)
    app.add_middleware(LoopbackHostGuard)
    app.include_router(
        build_ui_api_router(
            pots=pots, graph=graph, backend=backend, engine_client=engine_client
        ),
        prefix="/ui",
    )
    mount_ui_static(app)
    return app


__all__ = ["build_ui_app"]
