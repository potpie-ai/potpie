"""Serve the built React explorer SPA from the daemon under ``/ui``.

The frontend is a Vite build (``frontend/dist``) created during CLI install.
When the bundle is absent (e.g. a source checkout that hasn't run
``npm run build``), we mount a small placeholder page with build instructions
instead of 404-ing.

The shell and its assets are the one part of ``/ui`` that is served without a
credential: they have to load in order to run the handoff that *gets* one. They
carry no graph data — every byte of that comes from ``/ui/api``, which is
authenticated. The shell handler is also where a handoff code becomes a cookie,
because a redirect is the only way to take the code back out of the address bar.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from potpie.daemon.http.ui.auth import UiAuth

#: Query parameter carrying a single-use handoff code (see ``auth.py``).
HANDOFF_PARAM = "k"

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
<code>/ui/api/graph</code>, <code>/ui/api/catalog</code> — it needs the daemon
token, or the browser session <code>potpie ui</code> hands over.</p>
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
    served = index.is_file()

    # Registered before the mount below, so the shell paths reach this handler
    # (which knows about the handoff) while assets still come off the bundle.
    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def _ui_shell(request: Request) -> Response:
        code = request.query_params.get(HANDOFF_PARAM)
        if code is not None:
            return _redeem(request, code)
        return FileResponse(index) if served else HTMLResponse(_PLACEHOLDER)

    if served:
        app.mount("/ui", StaticFiles(directory=str(dist), html=True), name="ui")

    return served


def _redeem(request: Request, code: str) -> Response:
    """Spend a handoff code for a session cookie, then drop it from the URL.

    The redirect is the point: it leaves the browser holding an ``HttpOnly``
    cookie and a URL with no credential in it, so neither the address bar nor
    history keeps one. A code that is unknown, expired, or already spent sets no
    cookie and still lands on the clean URL — the page then loads and every
    ``/ui/api`` call answers 401 with what to do about it, which is a better
    account of the problem than an error page in place of the app.
    """
    auth = getattr(request.app.state, "ui_auth", None)
    session = auth.redeem_code(code) if isinstance(auth, UiAuth) else None
    clean = request.url.remove_query_params(HANDOFF_PARAM)
    target = clean.path + (f"?{clean.query}" if clean.query else "")
    # Relative on purpose: the browser stays on whatever host name it used to
    # get here (``localhost`` and ``127.0.0.1`` are different cookie origins).
    response = RedirectResponse(target, status_code=303)
    if session is not None:
        auth.attach_session(response, session)
    return response


__all__ = ["HANDOFF_PARAM", "frontend_dist_dir", "mount_ui_static"]
