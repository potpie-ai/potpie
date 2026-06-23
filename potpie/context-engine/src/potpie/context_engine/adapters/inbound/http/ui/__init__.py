"""Local graph-explorer UI inbound adapter.

A read-only browser surface served by the daemon: select the active pot and
explore the project-memory graph interactively. Talks to the same
``HostShell`` surfaces (``pots`` / ``graph`` / ``backend.inspection``) the CLI
uses — no new application logic, just an HTTP + SPA projection.
"""

from potpie.context_engine.adapters.inbound.http.ui.router import build_ui_api_router
from potpie.context_engine.adapters.inbound.http.ui.static import frontend_dist_dir, mount_ui_static

__all__ = ["build_ui_api_router", "frontend_dist_dir", "mount_ui_static"]
