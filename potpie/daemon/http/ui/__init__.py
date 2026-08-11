"""Local graph-explorer UI inbound adapter.

A read-only browser surface served by the daemon: select the active pot and
explore the project-memory graph interactively. Talks to the same
``HostShell`` surfaces (``pots`` / ``graph`` / ``backend.inspection``) the CLI
uses — no new application logic, just an HTTP + SPA projection.

The JSON API takes the same credential as ``/rpc``; the SPA shell trades a
handoff code for a session cookie so a browser can hold one (``auth.py``).
"""

from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import build_ui_api_router
from potpie.daemon.http.ui.static import frontend_dist_dir, mount_ui_static

__all__ = [
    "UiAuth",
    "build_ui_api_router",
    "frontend_dist_dir",
    "mount_ui_static",
]
