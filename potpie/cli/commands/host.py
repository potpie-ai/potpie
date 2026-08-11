"""``potpie host`` — which context-engines this CLI knows about.

The local daemon needs no configuration; a managed context-graph service needs
an address and, unless its auth is disabled, a token. That pair is stored under
its own name in the credentials file rather than in the Potpie *account*
fields, because the account API and a self-hosted context-graph service are
different things that may live at different addresses (see
``potpie.cli.hosts.managed_endpoint``).
"""

from __future__ import annotations

from typing import Any

import typer

from potpie.cli import hosts
from potpie.cli.commands._common import contract, emit, fail

host_app = typer.Typer(
    help="Context-engine hosts: the local daemon and a managed service."
)


@host_app.command("list")
def host_list() -> None:
    """Show each host, whether it is configured, and which one is active."""
    with contract():
        endpoint = hosts.managed_endpoint()
        # A configured endpoint that is not a usable address reads as "not
        # configured" to every router, which on its own is the same silence the
        # degrade note below exists to break.
        problem = hosts.managed_endpoint_problem()
        active = hosts.current_origin()
        persisted = hosts.persisted_origin()
        # What was asked for, which is the persisted pointer only when nothing
        # overrode it. The degrade note below has to name the origin that was
        # actually selected, or it describes a host nobody chose.
        selected = hosts.selected_origin()
        degraded = hosts.origin_degraded()
        rows = [
            {
                "origin": hosts.LOCAL,
                "configured": True,
                "endpoint": str(hosts.home_dir()),
                "problem": None,
                "active": active == hosts.LOCAL,
            },
            {
                "origin": hosts.MANAGED,
                "configured": endpoint is not None,
                "endpoint": endpoint[0] if endpoint else None,
                "problem": problem,
                "active": active == hosts.MANAGED,
            },
        ]
        lines = []
        for row in rows:
            marker = "*" if row["active"] else " "
            if row["problem"]:
                where = "(misconfigured, so nothing is routed to it)"
            else:
                where = (
                    row["endpoint"] or "(not configured — run 'potpie host set <url>')"
                )
            lines.append(f"{marker} {row['origin']:<8} {where}")
        if problem:
            lines.append(f"! {problem}")
        if degraded:
            # The one place the silent degrade in hosts.current_origin() is
            # visible. Without it the selected host is `managed`, every command
            # reads the local graph, and both facts exit 0 with nothing said.
            lines.append(
                f"! selected host is '{selected}' but none is configured; "
                f"commands are running against '{active}'"
            )
        emit(
            {
                "hosts": rows,
                "active_origin": active,
                "selected_origin": selected,
                "persisted_origin": persisted,
                "degraded": degraded,
                "problem": problem,
            },
            human="\n".join(lines),
        )


@host_app.command("set")
def host_set(
    url: str = typer.Argument(
        ..., help="Base URL of the managed service, e.g. http://127.0.0.1:8090"
    ),
    token: str = typer.Option(
        None,
        "--token",
        help=(
            "API key the service expects. Omit when the service runs with auth "
            "disabled; pass --token '' to clear a stored one."
        ),
    ),
    check: bool = typer.Option(
        True,
        "--check/--no-check",
        help="Verify the endpoint answers before saving.",
    ),
) -> None:
    """Point this CLI at a managed context-graph service."""
    with contract():
        base_url = _validated_base_url(url)
        token = _resolved_token(token)
        pots = _probe(base_url, token) if check else None
        # Written only once there is nothing left that can refuse it.
        # set_managed_endpoint replaces the stored url *and* token, so the old
        # write-first-clear-on-failure order turned a typo into the permanent
        # loss of a working endpoint and a token nobody else holds a copy of.
        hosts.set_managed_endpoint(base_url, token)
        payload: dict[str, Any] = {
            "origin": hosts.MANAGED,
            "endpoint": base_url,
            "checked": check,
        }
        human = f"managed host → {base_url}"
        if pots is not None:
            payload["pots"] = len(pots)
            human = f"{human} ({len(pots)} pots visible)"
        # The write succeeded and is still not what the next command will use:
        # POTPIE_MANAGED_URL outranks the file in hosts._resolve_managed(). Said
        # at the point of the write because that is where the belief forms —
        # `managed host → <url>` at exit 0 while every later command talks to the
        # environment's host is indistinguishable from writing to the wrong file.
        shadow = hosts.managed_env_override()
        if shadow:
            payload["shadowed_by_env"] = shadow
            human = (
                f"{human}\n! POTPIE_MANAGED_URL={shadow} is set and outranks the "
                "stored host; commands will use it until it is unset"
            )
        else:
            payload["shadowed_by_env"] = None
        emit(payload, human=human)


@host_app.command("clear")
def host_clear() -> None:
    """Forget the managed service; the CLI falls back to the local daemon."""
    with contract():
        salvaged = hosts.clear_managed_endpoint()
        # The active-origin pointer is left alone on purpose: current_origin()
        # already degrades to local when managed is unconfigured, so clearing
        # is reversible by re-running 'host set' without losing the selection.
        human = "managed host cleared"
        if salvaged is not None:
            # This command is the CLI's way out of an unreadable registry, so it
            # says what it replaced and where the old bytes went; every other
            # command still refuses to touch that file.
            human = f"{human} (unreadable registry replaced; old bytes at {salvaged})"
        emit(
            {
                "origin": hosts.MANAGED,
                "endpoint": None,
                "salvaged_registry": str(salvaged) if salvaged else None,
            },
            human=human,
        )


@host_app.command("use")
def host_use(origin: str = typer.Argument(..., metavar="local|managed")) -> None:
    """Make a host active without naming a pot."""
    with contract():
        origin = origin.strip().lower()
        try:
            hosts.require_origin(origin)
        except ValueError as exc:
            fail(
                code="validation_error",
                message=str(exc),
                next_action="run 'potpie host list'",
            )
        if origin == hosts.MANAGED and hosts.managed_endpoint() is None:
            fail(
                code="validation_error",
                message="No managed host is configured.",
                next_action="run 'potpie host set <url>'",
            )
        hosts.set_persisted_origin(origin)
        emit({"active_origin": origin}, human=f"active host → {origin}")


def _validated_base_url(url: str) -> str:
    """Normalize ``url`` or refuse it, before any host is built or written.

    ``--check`` decides whether the service is *reachable*; this decides whether
    the address is an address at all, which is a different answer with a
    different fix — and one that ``--no-check`` used to skip entirely, storing
    endpoints that could never be used and reporting success.
    """
    try:
        return hosts.normalize_managed_url(url)
    except ValueError as exc:
        fail(
            code="validation_error",
            message=str(exc),
            next_action="pass a full base URL, e.g. 'potpie host set http://127.0.0.1:8090'",
        )


def _resolved_token(token: str | None) -> str:
    """The token to store, refusing the one case where omitting one destroys one.

    Omitting ``--token`` means "this service has auth disabled" on a first setup
    and meant "erase the key I pasted in once" on every run after — identical
    keystrokes, and the second is recoverable from nowhere, since ``host set``
    replaces the url and the token together. So omission is only an answer while
    there is nothing to lose; past that the user says which of the two they meant.
    """
    if token is not None:
        # Normalized here, beside the URL, because this is the value the probe
        # below is handed *and* the value that gets stored. While only the read
        # path stripped it, `--token 'k3y '` was probed as `Bearer k3y ` and sent
        # as `Bearer k3y` by every command afterwards — a key refused at the door
        # in the one form that would have worked — and a whitespace-only token
        # probed as `Bearer   ` and then went out as the auth-disabled
        # placeholder.
        return hosts.normalize_managed_token(token)
    if not hosts.stored_managed_token():
        return ""
    fail(
        code="validation_error",
        message=(
            "A managed-host token is already stored, and 'host set' replaces the "
            "url and the token together."
        ),
        next_action=(
            "pass --token <key> to replace it, or --token '' to clear it deliberately"
        ),
    )


def _probe(base_url: str, token: str) -> list[Any]:
    """List pots on ``(base_url, token)`` to prove the pair works.

    Built from the pair being offered rather than from stored state — that is
    the whole point of probing first, and it also means the endpoint named in
    the error is the one that failed rather than whatever happened to be saved.
    """
    try:
        return list(hosts.build_managed_host(base_url, token).pots.list_pots())
    except Exception as exc:  # noqa: BLE001 - anything at all means "do not store"
        fail(
            code="unavailable",
            message=f"The managed host at {base_url} did not answer: {exc}",
            next_action=(
                "check the service is running and that --token is the key it "
                "expects, then run 'potpie host set <url>' again"
            ),
        )


__all__ = ["host_app"]
