"""``potpie ui`` → open the local graph-explorer served by the daemon.

A read-only browser surface to select the active pot and explore the
project-memory graph interactively. The page + its JSON API are served by the
daemon at ``/ui`` (loopback only); this command just makes sure the daemon is
up, then points you (and your browser) at the right URL.
"""

from __future__ import annotations

import webbrowser
from urllib.parse import urlencode

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_host,
    resolve_pot_id,
)


def ui_command(
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the explorer in your browser."
    ),
    pot: str = typer.Option(
        None,
        "--pot",
        help="Open the explorer against a specific pot id/name.",
    ),
) -> None:
    """Launch the local graph-explorer UI (served by the daemon)."""
    with contract():
        host = get_host()
        # Bring the detached daemon up if needed (in-process host is a no-op).
        try:
            host.daemon.ensure()
        except Exception:  # noqa: BLE001 — fall through to the discovery check
            pass
        disc = host.daemon.discovery()
        if not disc or not disc.get("base_url"):
            fail(
                code="daemon_unavailable",
                message="Potpie daemon is not running, so the UI can't be served.",
                next_action="run 'potpie setup' (or 'potpie daemon restart'), then 'potpie ui'",
            )
            return
        base = str(disc["base_url"]).rstrip("/")
        pot_id = resolve_pot_id(host, pot) if pot else None
        query = f"?{urlencode({'pot': pot_id})}" if pot_id else ""
        url = f"{base}/ui{query}"
        warning = _probe_ui(base)
        if open_browser and warning is None:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass
        lines = [f"Potpie Graph Explorer → {url}"]
        if warning:
            lines.append(f"  ! {warning}")
        elif open_browser:
            lines.append("  (opening in your browser…)")
        emit({"url": url, "pot_id": pot_id, "warning": warning}, human="\n".join(lines))


def _probe_ui(base: str) -> str | None:
    """Return a warning if the running daemon doesn't yet serve the UI."""
    import httpx

    try:
        resp = httpx.get(f"{base}/ui/api/pots", timeout=3.0)
    except Exception:  # noqa: BLE001 — daemon may still be booting
        return None
    if resp.status_code == 404:
        return "this daemon predates the UI — run 'potpie daemon restart' to enable it."
    return None


def register(app: typer.Typer) -> None:
    app.command("ui")(ui_command)


__all__ = ["register", "ui_command"]
