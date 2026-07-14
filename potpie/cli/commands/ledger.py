"""Event Ledger commands routed through ``PotpieRuntime.engine.ledger``.

The ledger is a separate source-event service. ``ledger query`` and
``ledger pull`` inspect event history only; graph updates are intentionally left
to ``context_record`` or the explicit ``graph propose`` / ``graph commit`` path.
"""

from __future__ import annotations

from datetime import datetime

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_cli_runtime,
    resolve_pot_id,
)
from potpie.runtime.async_bridge import run_sync
from potpie.runtime.contracts import (
    LedgerPullRequest,
    LedgerQueryRequest,
    LedgerSourcesRequest,
    LedgerStatusRequest,
)

ledger_app = typer.Typer(help="Event Ledger binding + query/pull/reconcile.")

_BINDINGS = {
    "managed": "managed",
    "self-hosted": "self_hosted",
    "self_hosted": "self_hosted",
}


def _parse_instant(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(
            code="validation_error",
            message=f"invalid ISO timestamp '{value}'",
            next_action="use an ISO-8601 instant, e.g. 2026-06-01T00:00:00Z",
        )


@ledger_app.command("status")
def ledger_status() -> None:
    with contract():
        runtime = get_cli_runtime()
        health = run_sync(lambda: runtime.engine.ledger.status(LedgerStatusRequest()))
        emit(
            {
                "available": health.available,
                "binding": health.binding,
                "detail": health.detail,
            },
            human=f"ledger: binding={health.binding} available={health.available}"
            + (f" ({health.detail})" if health.detail else ""),
        )


@ledger_app.command("sources")
def ledger_sources(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        runtime = get_cli_runtime()
        pot_id = resolve_pot_id(runtime, pot)
        sources = list(
            run_sync(
                lambda: runtime.engine.ledger.sources(
                    LedgerSourcesRequest(pot_id=pot_id)
                )
            ).items
        )
        emit(
            {"sources": [{"id": s.source_id, "provider": s.provider} for s in sources]},
            human="\n".join(f"  {s.provider}: {s.source_id}" for s in sources)
            or "(no ledger sources)",
        )


@ledger_app.command("query")
def ledger_query(
    source: str = typer.Option(None, "--source"),
    type_: str = typer.Option(None, "--type", help="Normalized event kind filter."),
    since: str = typer.Option(None, "--since", help="ISO instant lower bound."),
    until: str = typer.Option(None, "--until", help="ISO instant upper bound."),
    limit: int = typer.Option(100, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Inspect ledger event history (read-only; does not advance the cursor)."""
    with contract():
        runtime = get_cli_runtime()
        pot_id = resolve_pot_id(runtime, pot)
        page = run_sync(
            lambda: runtime.engine.ledger.query(
                LedgerQueryRequest(
                    pot_id=pot_id,
                    source_id=source,
                    kind=type_,
                    since=_parse_instant(since),
                    until=_parse_instant(until),
                    limit=limit,
                )
            )
        )
        emit(
            {
                "events": [
                    {
                        "id": e.event_id,
                        "source": e.source_id,
                        "provider": e.provider,
                        "kind": e.kind,
                    }
                    for e in page.events
                ],
                "has_more": page.has_more,
            },
            human="\n".join(
                f"  {e.provider}/{e.kind}: {e.event_id}" for e in page.events
            )
            or "(no matching events)",
        )


@ledger_app.command("use")
def ledger_use(
    binding: str = typer.Argument(..., help="managed | self-hosted"),
    url: str = typer.Argument(None, help="Ledger URL (required for self-hosted)."),
    org: str = typer.Option(None, "--org"),
) -> None:
    """Bind a managed or self-hosted Event Ledger for the local graph."""
    with contract():
        normalized = _BINDINGS.get(binding.strip().lower())
        if normalized is None:
            fail(
                code="validation_error",
                message=f"unknown ledger binding '{binding}'",
                next_action="use 'managed' or 'self-hosted <url>'",
            )
        if normalized == "self_hosted" and not url:
            fail(
                code="validation_error",
                message="self-hosted ledger requires a URL",
                next_action="run 'potpie ledger use self-hosted <url>'",
            )
        config = get_cli_runtime().config
        config.set("ledger.binding", normalized)
        if org:
            config.set("ledger.org", org)
        if url:
            config.set("ledger.url", url)
        emit(
            {
                "binding": normalized,
                "url": url,
                "org": org,
                "persisted": True,
                "active": False,
            },
            human=(
                f"ledger binding recorded: {normalized}"
                + (f" ({url})" if url else "")
                + " — runtime ledger rebinding lands with managed routing (HU3)"
            ),
        )


@ledger_app.command("disconnect")
def ledger_disconnect() -> None:
    """Clear the Event Ledger binding."""
    with contract():
        get_cli_runtime().config.set("ledger.binding", "none")
        emit({"binding": "none", "persisted": True}, human="ledger disconnected")


@ledger_app.command("pull")
def ledger_pull(
    source: str = typer.Option(..., "--source"),
    filter_: str = typer.Option(
        None, "--filter", help="Reserved: server-side event filter expression."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with contract():
        runtime = get_cli_runtime()
        pot_id = resolve_pot_id(runtime, pot)
        page = run_sync(
            lambda: runtime.engine.ledger.pull(
                LedgerPullRequest(pot_id=pot_id, source_id=source)
            )
        )
        emit(
            {
                "pulled": len(page.events),
                "has_more": page.has_more,
            },
            human=f"pulled {len(page.events)} events (read-only)",
        )


__all__ = ["ledger_app"]
