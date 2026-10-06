"""Local graph-explorer UI inbound adapter.

A read-only browser surface served by the daemon: select the active pot and
explore the project-memory graph interactively. Talks to the same explicit pot,
graph, and backend services the CLI uses — no new application logic, just an
HTTP + SPA projection.

The JSON API takes the daemon's bearer token; the SPA shell trades a single-use
handoff code for a session cookie so a browser can hold a credential without
ever seeing the token (``auth.py``).
"""

from potpie.daemon.http.ui.app import build_ui_app
from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import build_ui_api_router
from potpie.daemon.http.ui.static import frontend_dist_dir, mount_ui_static

__all__ = [
    "UiAuth",
    "build_ui_api_router",
    "build_ui_app",
    "frontend_dist_dir",
    "mount_ui_static",
]
