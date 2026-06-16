"""Serve the built React explorer SPA from the daemon under ``/ui``.

The frontend is a Vite build (``frontend/dist``) committed alongside the source
so an installed daemon can serve the UI with no Node toolchain at runtime. When
the bundle is absent (e.g. a source checkout that hasn't run ``npm run build``)
we mount a small placeholder page with build instructions instead of 404-ing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_PLACEHOLDER = """<!doctype html>
<html><head><meta charset="utf-8"><title>Potpie Graph Explorer</title>
<style>body{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:40rem;
margin:4rem auto;padding:0 1rem;color:#222}code{background:#f3f3f5;padding:.15em .4em;
border-radius:4px}</style></head><body>
<h1>Potpie Graph Explorer</h1>
<p>The UI bundle has not been built yet. From the repo root, run:</p>
<pre><code>make cli-install
# or: make ui-build
potpie daemon restart</code></pre>
<p>The JSON API is live now at <code>/ui/api/pots</code>,
<code>/ui/api/graph</code>, <code>/ui/api/catalog</code>.</p>
</body></html>"""


def frontend_dist_dir() -> Path:
    """Absolute path to the built SPA (may not exist in a fresh checkout)."""
    return Path(__file__).resolve().parent / "frontend" / "dist"


def mount_ui_static(app: FastAPI) -> bool:
    """Mount the built SPA at ``/ui``; return True if a real bundle was served.

    Must be called *after* the ``/ui/api`` router is included so API routes
    win over the catch-all static mount.
    """
    dist = frontend_dist_dir()
    index = dist / "index.html"
    if index.is_file():
        app.mount("/ui", StaticFiles(directory=str(dist), html=True), name="ui")
        return True

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def _ui_placeholder() -> Any:  # pragma: no cover - dev convenience
        return HTMLResponse(_PLACEHOLDER)

    return False


__all__ = ["frontend_dist_dir", "mount_ui_static"]
