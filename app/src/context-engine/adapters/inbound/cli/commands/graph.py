"""Graph + backend commands → ``HostShell.graph`` and the active ``GraphBackend``.

CLI code never touches a store directly; everything goes through the capability
ports. Unbuilt projections (semantic/inspection/analytics/snapshot on profiles
that lack them) surface as the structured not-implemented contract.
"""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import (
    contract,
    emit,
    get_host,
    resolve_pot_id,
)

graph_app = typer.Typer(help="Graph reads/admin via capability ports.")
backend_app = typer.Typer(help="GraphBackend profile selection + health.")


@graph_app.command("status")
def graph_status(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        dp = host.graph.data_plane_status(pot_id)
        emit(
            {
                "backend": dp.backend_profile,
                "ready": dp.backend_ready,
                "counts": dict(dp.counts),
                "freshness": dict(dp.freshness),
                "quality": dict(dp.quality),
            },
            human=f"backend={dp.backend_profile} ready={dp.backend_ready} counts={dict(dp.counts)}",
        )


@graph_app.command("inspect")
def graph_inspect(
    entity_key: str,
    depth: int = typer.Option(2, "--depth"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        sl = host.backend.inspection.neighborhood(
            pot_id=pot_id, entity_key=entity_key, depth=depth
        )
        emit(
            {
                "nodes": [{"key": n.key, "labels": list(n.labels)} for n in sl.nodes],
                "edges": [
                    {"predicate": e.predicate, "from": e.from_key, "to": e.to_key}
                    for e in sl.edges
                ],
            },
            human=f"{len(sl.nodes)} nodes, {len(sl.edges)} edges around {entity_key}",
        )


@graph_app.command("export")
def graph_export(file: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        manifest = host.backend.snapshot.export(pot_id=pot_id, destination=file)
        emit(
            {"location": manifest.location, "claims": manifest.claim_count},
            human=f"exported {manifest.claim_count} claims → {manifest.location}",
        )


@graph_app.command("import")
def graph_import(file: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        manifest = host.backend.snapshot.import_(pot_id=pot_id, source=file)
        emit(
            {"location": manifest.location, "claims": manifest.claim_count},
            human=f"imported {manifest.claim_count} claims from {manifest.location}",
        )


@graph_app.command("repair")
def graph_repair(
    semantic_index: bool = typer.Option(False, "--semantic-index"),
    all_: bool = typer.Option(False, "--all"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        targets = [] if all_ else (["semantic_index"] if semantic_index else [])
        report = host.backend.analytics.repair(pot_id, targets=targets)
        emit(
            {"targets": list(report.targets), "repaired": dict(report.repaired)},
            human=report.detail or f"repaired {dict(report.repaired)}",
        )


@backend_app.command("list")
def backend_list() -> None:
    with contract():
        from adapters.outbound.graph.backends import KNOWN_PROFILES

        active = get_host().backend.profile
        emit(
            {"profiles": list(KNOWN_PROFILES), "active": active},
            human="\n".join(
                f"{'*' if p == active else ' '} {p}" for p in KNOWN_PROFILES
            ),
        )


@backend_app.command("status")
def backend_status() -> None:
    with contract():
        host = get_host()
        caps = host.backend.capabilities()
        emit(
            {"profile": host.backend.profile, "capabilities": list(caps.implemented())},
            human=f"{host.backend.profile}: {', '.join(caps.implemented())}",
        )


@backend_app.command("use")
def backend_use(profile: str) -> None:
    # TODO(stage-N): persist the selected profile to local config; the host
    # rebuilds per process from $CONTEXT_ENGINE_BACKEND today.
    with contract():
        emit(
            {"profile": profile, "persisted": False},
            human=f"(advisory) set CONTEXT_ENGINE_BACKEND={profile}; persistence is TODO",
        )


@backend_app.command("doctor")
def backend_doctor() -> None:
    with contract():
        host = get_host()
        pot = host.pots.active_pot()
        readiness = host.backend.mutation.readiness(pot.pot_id if pot else "")
        emit(
            {
                "profile": readiness.profile,
                "ready": readiness.ready,
                "capability_ready": dict(readiness.capability_ready),
                "detail": readiness.detail,
            },
            human=f"{readiness.profile} ready={readiness.ready} {dict(readiness.capability_ready)}",
        )


__all__ = ["backend_app", "graph_app"]
