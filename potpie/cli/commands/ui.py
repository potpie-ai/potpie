"""``potpie ui`` → open the graph-explorer served by the local daemon.

A read-only browser surface to select the active pot and explore the
project-memory graph interactively. The page + its JSON API are served by the
daemon at ``/ui`` (loopback only); this command just makes sure the daemon is
up, then points you (and your browser) at the right URL.

The *serving* host is always the local daemon, even when the pot you are opening
lives on a managed service. A managed endpoint speaks the RPC surface and needs
a bearer token, which a browser navigation cannot supply and which has no
business sitting in a URL bar; the daemon holds the token and proxies the reads,
so the browser never leaves loopback. Which graph you are looking at is carried
in ``?host=``, exactly as ``--host`` carries it on the command line.

The *local* daemon needs a credential too — its UI API is on loopback, which
every other process on this machine is also on. This command is the one that can
read the daemon token out of ``discovery.json``, so it spends the token on a
single-use handoff code and puts that in the opened URL; the daemon swaps the
code for an HttpOnly cookie and redirects, leaving no credential in the browser's
address bar or history.
"""

from __future__ import annotations

import webbrowser
from urllib.parse import urlencode

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_host_for,
    resolve_pot_id,
)


def ui_command(
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the explorer in your browser."
    ),
    pot: str = typer.Option(
        None,
        "--pot",
        help="Open the explorer against a specific pot ref, e.g. 'api' or 'managed:api'.",
    ),
) -> None:
    """Launch the graph-explorer UI (served by the local daemon)."""
    with contract():
        from potpie.cli import hosts

        # Resolve the pot before the daemon: a qualified ref like 'managed:api'
        # moves the target origin, and the URL has to name the host the pot id
        # was actually looked up on, or the explorer reads an id from one graph
        # against the other.
        target = hosts.current_origin()
        pot_id = resolve_pot_id(get_host_for(target), pot) if pot else None
        if pot:
            target = hosts.current_origin()

        # The daemon serves the page regardless of which host owns the pot.
        local = get_host_for(hosts.LOCAL)
        # Bring the detached daemon up if needed (in-process host is a no-op).
        try:
            local.daemon.ensure()
        except Exception:  # noqa: BLE001 — fall through to the discovery check
            pass
        disc = local.daemon.discovery()
        if not disc or not disc.get("base_url"):
            fail(
                code="daemon_unavailable",
                message="Potpie daemon is not running, so the UI can't be served.",
                next_action="run 'potpie setup' (or 'potpie daemon restart'), then 'potpie ui'",
            )
            return
        base = str(disc["base_url"]).rstrip("/")
        params = {"host": target}
        if pot_id:
            params["pot"] = pot_id
        warning = _probe_ui(disc)
        if warning is None:
            code, warning = _handoff_code(disc)
            if code:
                # Single-use and short-lived: whoever opens this URL first gets
                # the session, and the code is dead by the time it lands in
                # history.
                params["k"] = code
        url = f"{base}/ui?{urlencode(params)}"
        if open_browser and warning is None:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass
        lines = [
            f"Potpie Graph Explorer → {url}",
            f"  host: {hosts.origin_label(target)}",
        ]
        if warning:
            lines.append(f"  ! {warning}")
        elif open_browser:
            lines.append("  (opening in your browser…)")
        emit(
            {"url": url, "pot_id": pot_id, "origin": target, "warning": warning},
            human="\n".join(lines),
        )


def _endpoint(disc: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Base URL and auth header for the daemon a discovery record names.

    Address and credential travel together: the UI API takes the same token
    ``/rpc`` does, and discovery is where both come from.
    """
    base = str(disc.get("base_url") or "").rstrip("/")
    token = str(disc.get("token") or "")
    return base, {"Authorization": f"Bearer {token}"} if token else {}


def _probe_ui(disc: dict[str, str]) -> str | None:
    """Return a warning if the running daemon can't serve what we're asking for.

    The daemon is long-lived: after an upgrade the process still running is the
    one from before it, so these checks are about a *stale* daemon rather than a
    broken one. The host-routing one matters most on a managed host — an older
    daemon answers ``/ui/api/pots`` perfectly well, just with its own pots only,
    and the managed graph would silently be missing rather than reported.
    """
    import httpx

    base, headers = _endpoint(disc)
    try:
        resp = httpx.get(f"{base}/ui/api/pots", headers=headers, timeout=3.0)
    except Exception:  # noqa: BLE001 — daemon may still be booting
        return None
    if resp.status_code == 404:
        return "this daemon predates the UI — run 'potpie daemon restart' to enable it."
    if resp.status_code == 401:
        # Discovery and the daemon disagree about the token, so nothing this
        # command hands the browser would be accepted either.
        return (
            "the daemon rejected this machine's daemon token — run "
            "'potpie daemon restart'."
        )
    try:
        knows_hosts = "active_origin" in resp.json()
    except Exception:  # noqa: BLE001
        return None
    if not knows_hosts:
        return (
            "this daemon predates host routing, so the explorer will only show "
            "local pots — run 'potpie daemon restart'."
        )
    return None


def _handoff_code(disc: dict[str, str]) -> tuple[str | None, str | None]:
    """``(code, warning)`` for the browser session this URL should carry.

    The code is what lets the browser authenticate without ever holding the
    daemon token, so *no code* means the explorer opens onto a wall of 401s.
    Exactly one silence is honest: a daemon that has never heard of the route
    predates the gate and still serves its API open, so the page works. Every
    other way of coming back empty-handed — refused, unreachable, or a 200
    carrying nothing usable — is reported, because the alternative is this
    command announcing a handoff that did not happen.
    """
    import httpx

    base, headers = _endpoint(disc)
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
