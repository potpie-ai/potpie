"""Pot + source commands → ``HostShell.pots`` (PotManagementService)."""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import (
    contract,
    emit,
    get_host,
    resolve_pot_id,
)
from domain.errors import CapabilityNotImplemented

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
source_app = typer.Typer(help="Sources registered to a pot.")


@pot_app.command("list")
def pot_list(
    local: bool = typer.Option(False, "--local", help="Local-origin pots only (default)."),
    managed: bool = typer.Option(False, "--managed", help="Managed-origin pots only."),
    all_: bool = typer.Option(False, "--all", help="Local + managed pots."),
) -> None:
    with contract():
        # Managed-origin listing needs login + managed routing (HU3). '--managed'
        # alone is the structured not-implemented contract; '--all' shows locals
        # and flags managed as pending so it never crashes.
        if managed and not all_:
            raise CapabilityNotImplemented(
                "host.pots.list_managed",
                detail="managed pot listing is not implemented",
                recommended_next_action="run 'potpie login'; managed routing lands in HU3",
            )
        pots = get_host().pots.list_pots()
        payload: dict[str, object] = {
            "pots": [
                {"id": p.pot_id, "name": p.name, "active": p.active, "origin": "local"}
                for p in pots
            ],
        }
        human_lines = [
            f"{'*' if p.active else ' '} {p.name} ({p.pot_id}) [local]" for p in pots
        ] or ["(no pots)"]
        if all_:
            payload["managed_pending"] = True
            human_lines.append("  (managed pots require 'potpie login' — HU3)")
        emit(payload, human="\n".join(human_lines))


@pot_app.command("info")
def pot_info() -> None:
    with contract():
        active = get_host().pots.active_pot()
        if active is None:
            emit({"active_pot": None}, human="(no active pot)")
            return
        emit(
            {"active_pot": {"id": active.pot_id, "name": active.name}},
            human=f"active: {active.name} ({active.pot_id})",
        )


@pot_app.command("create")
def pot_create(
    name: str,
    repo: str = typer.Option(None, "--repo"),
    use: bool = typer.Option(False, "--use"),
) -> None:
    with contract():
        pot = get_host().pots.create_pot(name=name, repo=repo, use=use)
        emit(
            {"id": pot.pot_id, "name": pot.name, "active": pot.active},
            human=f"created pot '{pot.name}' ({pot.pot_id}){' [active]' if pot.active else ''}",
        )


@pot_app.command("use")
def pot_use(ref: str) -> None:
    with contract():
        pot = get_host().pots.use_pot(ref=ref)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"active pot → {pot.name}")


@pot_app.command("rename")
def pot_rename(ref: str, new_name: str) -> None:
    with contract():
        pot = get_host().pots.rename_pot(ref=ref, new_name=new_name)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"renamed → {pot.name}")


@pot_app.command("reset")
def pot_reset(
    ref: str = typer.Argument(None),
    confirm: bool = typer.Option(False, "--confirm"),
) -> None:
    with contract():
        host = get_host()
        target = ref or resolve_pot_id(host)
        pot = host.pots.reset_pot(ref=target, confirm=confirm)
        emit({"id": pot.pot_id, "reset": True}, human=f"reset graph state for '{pot.name}'")


@pot_app.command("archive")
def pot_archive(ref: str) -> None:
    with contract():
        pot = get_host().pots.archive_pot(ref=ref)
        emit({"id": pot.pot_id, "archived": True}, human=f"archived '{pot.name}'")


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(..., help="repo | github | linear | …"),
    location: str = typer.Argument(...),
    name: str = typer.Option(None, "--name"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        src = host.pots.add_source(pot_id=pot_id, kind=kind, location=location, name=name)
        emit(
            {"source_id": src.source_id, "kind": src.kind, "name": src.name},
            human=f"added source {src.kind}:{src.name} ({src.source_id})",
        )


@source_app.command("list")
def source_list(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        sources = host.pots.list_sources(pot_id=pot_id)
        emit(
            {"sources": [{"id": s.source_id, "kind": s.kind, "name": s.name} for s in sources]},
            human="\n".join(f"  {s.kind}: {s.name} ({s.source_id})" for s in sources)
            or "(no sources)",
        )


@source_app.command("status")
def source_status(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        src = host.pots.source_status(pot_id=pot_id, source_id=source_id)
        emit(
            {"id": src.source_id, "status": src.status, "name": src.name},
            human=f"{src.name}: {src.status}",
        )


@source_app.command("remove")
def source_remove(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        host.pots.remove_source(pot_id=pot_id, source_id=source_id)
        emit({"removed": source_id}, human=f"removed source {source_id}")


__all__ = ["pot_app", "source_app"]
