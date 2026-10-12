"""``potpie host``: which context hosts this CLI knows about.

The local daemon needs no configuration; a managed context-graph service needs
an address and a token (a loopback service with auth disabled may have none).
The pair lives in ``<home>/cli_hosts.json`` (see
:mod:`potpie.config.host_registry`). A key never travels on argv: ``--token -``
reads it from stdin.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from typing import Any, NoReturn

import click
import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    contract,
    emit,
    fail,
)
from potpie.config import host_registry

host_app = typer.Typer(
    help="Context hosts: the local daemon and a managed context-graph service."
)

#: The module that owns the managed-host client. Imported lazily, and only by
#: ``host set --check``, so the registry commands work without it.
_REMOTE_MODULE = "potpie.runtime.remote"


@host_app.command("list")
def host_list() -> None:
    """Show each host, whether it is configured, and which one is active."""
    with contract():
        managed = host_registry.managed_host()
        # A configured endpoint that is not usable reads as "not configured" to
        # every router; the problem column is what keeps that from being silent.
        problem = host_registry.managed_endpoint_problem()
        active = host_registry.current_origin()
        persisted = host_registry.persisted_origin()
        selected = host_registry.selected_origin()
        degraded = host_registry.origin_degraded()
        rows = [
            {
                "origin": host_registry.LOCAL,
                "configured": True,
                "endpoint": str(host_registry.home_dir()),
                "problem": None,
                "active": active == host_registry.LOCAL,
                "allow_insecure_host": None,
            },
            {
                "origin": host_registry.MANAGED,
                "configured": managed is not None,
                "endpoint": managed.base_url if managed else None,
                "problem": problem,
                "active": active == host_registry.MANAGED,
                "allow_insecure_host": managed.allow_insecure_host if managed else None,
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
                if row["allow_insecure_host"]:
                    where = f"{where} (plain http, insecure opt-in)"
            lines.append(f"{marker} {row['origin']:<8} {where}")
        if problem:
            lines.append(f"! {problem}")
        if degraded:
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
        ..., help="Base URL of the managed service, e.g. https://your-host.example"
    ),
    token: str = typer.Option(
        None,
        "--token",
        help=(
            "'-' reads the API key from stdin; '' clears a stored one. A literal "
            "key is refused (it would be visible in the process list); "
            "POTPIE_MANAGED_TOKEN is the environment alternative."
        ),
    ),
    check: bool = typer.Option(
        True,
        "--check/--no-check",
        help="Verify the endpoint answers before saving.",
    ),
    allow_insecure: bool = typer.Option(
        False,
        "--allow-insecure",
        help=(
            "Accept plain http:// for this one non-loopback host[:port]. The API "
            "key then travels unencrypted."
        ),
    ),
) -> None:
    """Point this CLI at a managed context-graph service."""
    with contract():
        allow_insecure_host = (
            host_registry.insecure_host_key(url) if allow_insecure else None
        )
        base_url = validated_base_url(url, allow_insecure_host=allow_insecure_host)
        resolved = resolved_token(token, base_url)
        pots = _checked_pots(base_url, resolved, allow_insecure_host) if check else None
        # Written only once nothing is left that can refuse it: the write
        # replaces the url and the token together, so writing first would turn
        # a typo into the loss of a working endpoint and a token nobody else
        # holds a copy of.
        host_registry.set_managed_endpoint(
            base_url, resolved, allow_insecure_host=allow_insecure_host
        )
        # Every later command sends this key in the clear; say so once here
        # too (a no-op when the probe above already said it).
        host_registry.warn_insecure_transport(base_url)
        payload: dict[str, Any] = {
            "origin": host_registry.MANAGED,
            "endpoint": base_url,
            "checked": check,
            "allow_insecure_host": allow_insecure_host,
        }
        human = f"managed host → {base_url}"
        if pots is not None:
            payload["pots"] = len(pots)
            human = f"{human} ({len(pots)} pots visible)"
        # POTPIE_MANAGED_URL outranks the file, so the write landed and is still
        # not what the next command will use. Said where the belief forms.
        shadow = host_registry.managed_env_override()
        payload["shadowed_by_env"] = shadow
        if shadow:
            human = (
                f"{human}\n! POTPIE_MANAGED_URL={shadow} is set and outranks the "
                "stored host; commands will use it until it is unset"
            )
        emit(payload, human=human)


@host_app.command("clear")
def host_clear() -> None:
    """Forget the managed service; the CLI falls back to the local daemon."""
    with contract():
        salvaged = host_registry.clear_managed_endpoint()
        # The active-origin pointer is left alone: current_origin() degrades to
        # local while managed is unconfigured, so re-running 'host set' restores
        # the selection.
        human = "managed host cleared"
        if salvaged is not None:
            human = f"{human} (unreadable registry replaced; old bytes at {salvaged})"
        emit(
            {
                "origin": host_registry.MANAGED,
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
            host_registry.require_origin(origin)
        except ValueError as exc:
            fail(
                code="validation_error",
                message=str(exc),
                next_action="run 'potpie host list'",
            )
        if origin == host_registry.MANAGED and host_registry.managed_endpoint() is None:
            message, next_action = host_registry.managed_unconfigured_refusal()
            fail(code="validation_error", message=message, next_action=next_action)
        host_registry.set_persisted_origin(origin)
        emit({"active_origin": origin}, human=f"active host → {origin}")


def validated_base_url(url: str, *, allow_insecure_host: str | None = None) -> str:
    """Normalize ``url`` or refuse it, before any probe or write.

    ``--check`` decides whether the service is reachable; this decides whether
    the address is usable at all, which ``--no-check`` does not waive.
    """
    try:
        return host_registry.normalize_managed_url(
            url, allow_insecure_host=allow_insecure_host
        )
    except ValueError as exc:
        fail(
            code="validation_error",
            message=str(exc),
            next_action=(
                "pass a full https base URL, e.g. "
                "'potpie host set https://your-host.example'"
            ),
        )


def _refuse_literal_token() -> NoReturn:
    # The value itself is never echoed: it is the secret being protected.
    fail(
        code="validation_error",
        message=(
            "A literal --token value is refused: a key on the command line is "
            "visible in the process list and shell history. Pass --token - to "
            "read it from stdin, or set POTPIE_MANAGED_TOKEN."
        ),
        next_action=(
            "printf '%s\\n' \"$KEY\" | potpie host set <url> --token -  "
            "(or export POTPIE_MANAGED_URL and POTPIE_MANAGED_TOKEN)"
        ),
    )


def _token_from_stdin() -> str:
    """One line from stdin, or a hidden prompt when stdin is a terminal."""
    if sys.stdin is not None and sys.stdin.isatty():
        entered = click.prompt(
            "Managed host API key",
            hide_input=True,
            default="",
            show_default=False,
            err=True,
        )
        return host_registry.normalize_managed_token(str(entered))
    line = sys.stdin.readline() if sys.stdin is not None else ""
    return host_registry.normalize_managed_token(line)


def resolved_token(token: str | None, base_url: str) -> str:
    """The token to store, after the argv and destruction rules.

    - ``--token -`` reads the key from stdin; reading nothing is refused, so a
      broken pipe cannot silently clear a stored key.
    - ``--token ''`` (or whitespace) clears deliberately.
    - Any other value is refused: a key on argv is visible in ``ps``.
    - Omitting ``--token`` means "no token" on a first setup, but would erase a
      stored key on every later run, so it is refused while one is stored.

    An empty result is accepted only for a loopback host.
    """
    if token is None:
        if host_registry.stored_managed_token():
            fail(
                code="validation_error",
                message=(
                    "A managed-host token is already stored, and 'host set' "
                    "replaces the url and the token together."
                ),
                next_action=(
                    "pass --token - (key on stdin) to replace it, or --token '' "
                    "to clear it deliberately"
                ),
            )
        resolved = ""
    elif token == "-":
        resolved = _token_from_stdin()
        if not resolved:
            fail(
                code="validation_error",
                message="--token - read no key from stdin.",
                next_action=(
                    "write the key on stdin, e.g. printf '%s\\n' \"$KEY\" | "
                    "potpie host set <url> --token -; pass --token '' to clear "
                    "a stored key deliberately"
                ),
            )
    elif not host_registry.normalize_managed_token(token):
        resolved = ""
    else:
        _refuse_literal_token()
    problem = host_registry.managed_token_problem(base_url, resolved)
    if problem is not None:
        fail(
            code="validation_error",
            message=f"{problem[0].upper()}{problem[1:]}.",
            next_action=(
                "pass --token - and write the key on stdin, or set "
                "POTPIE_MANAGED_TOKEN (with POTPIE_MANAGED_URL)"
            ),
        )
    return resolved


class _ManagedClientMissing(Exception):
    """This installation has no ``potpie.runtime.remote`` to probe with."""


def _checked_pots(
    base_url: str, token: str, allow_insecure_host: str | None
) -> list[Any]:
    """Run the probe, turning any failure into "do not store"."""
    try:
        return _probe_managed(base_url, token, allow_insecure_host)
    except _ManagedClientMissing:
        fail(
            code="not_implemented",
            message=(
                "This potpie build has no managed-host client, so 'host set "
                "--check' cannot verify the endpoint."
            ),
            next_action="re-run with --no-check to save the endpoint without checking it",
            exit_code=EXIT_UNAVAILABLE,
        )
    except Exception as exc:  # noqa: BLE001 - anything at all means "do not store"
        fail(
            code="unavailable",
            message=f"The managed host at {base_url} did not answer: {exc}",
            next_action=(
                "check the service is running and that the key is the one it "
                "expects, then run 'potpie host set <url>' again"
            ),
            exit_code=EXIT_UNAVAILABLE,
        )


def _probe_managed(
    base_url: str, token: str, allow_insecure_host: str | None
) -> list[Any]:
    """Handshake and ``host.pot_list`` on an explicit, unsaved endpoint.

    The one call into the remote client, imported lazily so the registry
    commands work without it. Built from the pair being offered, never from
    stored state: probing first is what lets a failed check leave the stored
    pair untouched, and the error names the endpoint that failed.
    """
    try:
        remote = importlib.import_module(_REMOTE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _REMOTE_MODULE:
            raise
        raise _ManagedClientMissing(_REMOTE_MODULE) from exc
    # The probe sends the key: warn first when it goes out over plain http.
    host_registry.warn_insecure_transport(base_url)
    endpoint = remote.ManagedHostEndpoint(
        base_url=base_url, token=token, allow_insecure_host=allow_insecure_host
    )

    async def _probe() -> list[Any]:
        client = remote.RemoteHostClient(endpoint)
        try:
            await _settled(client.handshake())
            listing = await _settled(client.pot_list())
        finally:
            closer = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(closer):
                await _settled(closer())
        return list(getattr(listing, "pots", listing))

    return asyncio.run(_probe())


async def _settled(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = [
    "host_app",
    "resolved_token",
    "validated_base_url",
]
