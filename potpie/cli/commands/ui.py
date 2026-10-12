"""``potpie ui`` → open the local graph-explorer served by the daemon.

A read-only browser surface to select the active pot and explore the
project-memory graph interactively. The page + its JSON API are served by the
daemon at ``/ui`` (loopback only); this command makes sure the daemon is up,
then points you (and your browser) at the right URL.

The UI API needs a credential -- it is on loopback, which every other process
on this machine is also on. This command is the one that can read the daemon
token from the owner-only credential file, so it spends the token on a
single-use handoff code and puts that code in the URL it opens. The daemon
swaps the code for an HttpOnly cookie and redirects, leaving no credential in
the browser's address bar, and the token itself never reaches the browser.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_daemon_service,
    get_root_runtime,
    resolve_pot_id,
)

#: Query parameter the daemon's shell handler redeems (``daemon/http/ui``).
_HANDOFF_PARAM = "k"


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
        host = get_root_runtime()
        daemon = get_daemon_service(host)
        # Bring the detached daemon up if needed (in-process host is a no-op).
        try:
            daemon.ensure()
        except Exception:  # noqa: BLE001 — fall through to the discovery check
            pass
        daemon_status = daemon.status()
        if not daemon_status.get("ready") or not daemon_status.get("url"):
            fail(
                code="daemon_unavailable",
                message="Potpie daemon is not running, so the UI can't be served.",
                next_action="run 'potpie setup' (or 'potpie daemon restart'), then 'potpie ui'",
            )
            return
        base = str(daemon_status["url"]).rstrip("/")
        pot_id = resolve_pot_id(host, pot) if pot else None
        params: dict[str, str] = {}
        if pot_id:
            params["pot"] = pot_id
        headers = _auth_headers(daemon)
        warning = _probe_ui(base, headers)
        if warning is None:
            code, warning = _handoff_code(base, headers)
            if code:
                # Single-use and short-lived: whoever opens this URL first gets
                # the session, and the code is dead by the time it lands in
                # history. Callers that open the URL themselves (``--no-open``)
                # need it too, or their page would load onto a wall of 401s.
                params[_HANDOFF_PARAM] = code
        url = f"{base}/ui{'?' + urlencode(params) if params else ''}"
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


def _auth_headers(daemon: Any) -> dict[str, str]:
    """The bearer header for the daemon this command just checked on.

    Read from the owner-only credential file the daemon writes at boot -- the
    same secret the typed endpoint takes. No credential (an unreadable file)
    means no header: the daemon's 401 is then reported by ``_probe_ui``.
    """
    from potpie.config.local_paths import default_home
    from potpie.daemon.discovery import DaemonDiscoveryError, read_daemon_credential

    home = getattr(daemon, "home", None)
    try:
        token = read_daemon_credential(Path(home) if home else default_home())
    except (DaemonDiscoveryError, OSError):
        return {}
    return {"Authorization": f"Bearer {token}"}


def _probe_ui(base: str, headers: dict[str, str]) -> str | None:
    """Return a warning if the running daemon can't serve the explorer."""
    import httpx

    try:
        resp = httpx.get(f"{base}/ui/api/pots", headers=headers, timeout=3.0)
    except Exception:  # noqa: BLE001 — daemon may still be booting
        return None
    if resp.status_code == 404:
        return "this daemon predates the UI — run 'potpie daemon restart' to enable it."
    if resp.status_code == 401:
        # The credential file and the daemon disagree (or the file could not be
        # read), so nothing this command hands the browser would work either.
        return (
            "the daemon rejected this machine's daemon credential — run "
            "'potpie daemon restart'."
        )
    return None


def _handoff_code(base: str, headers: dict[str, str]) -> tuple[str | None, str | None]:
    """``(code, warning)`` for the browser session this URL should carry.

    The code is what lets the browser authenticate without ever holding the
    daemon token, so *no code* means the explorer opens onto 401s. Exactly one
    silence is honest: a daemon that has never heard of the route predates the
    gate and still serves its API open, so the page works. Every other way of
    coming back empty-handed -- refused, unreachable, or a 200 carrying nothing
    usable -- is reported, rather than announcing a handoff that did not happen.
    """
    import httpx

    try:
        resp = httpx.post(f"{base}/ui/api/handoff", headers=headers, timeout=3.0)
    except Exception as exc:  # noqa: BLE001 — every transport failure is "no session"
        return None, (
            "could not reach the daemon to start a browser session "
            f"({str(exc) or exc.__class__.__name__}) — it may still be starting; "
            "re-run 'potpie ui', or 'potpie daemon restart' if it persists."
        )
    if resp.status_code == 404:
        return None, None
    if resp.status_code != 200:
        return None, (
            "the daemon would not issue a browser session "
            f"(HTTP {resp.status_code}) — run 'potpie daemon restart'."
        )
    try:
        code = str(resp.json().get("code") or "")
    except Exception:  # noqa: BLE001 — an unreadable body is no code either
        code = ""
    if not code:
        return None, (
            "the daemon accepted the handoff but returned no session code — "
            "run 'potpie daemon restart'."
        )
    return code, None


def register(app: typer.Typer) -> None:
    app.command("ui")(ui_command)


__all__ = ["register", "ui_command"]
