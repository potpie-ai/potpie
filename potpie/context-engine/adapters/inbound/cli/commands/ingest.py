"""Ingestion commands (scanner-driven).

``ingest scan`` walks the working tree, runs every registered deterministic
config scanner (CODEOWNERS, dependency manifests, Kubernetes/Helm, OpenAPI),
and writes the resulting claims through ``HostShell.ingest`` → the backend's
mutation port — the same canonical path ``record`` uses. Run history is not yet
persisted on the local profile (``status``/``runs`` are stubs).
"""

from __future__ import annotations

import uuid

import typer

from adapters.inbound.cli.commands._common import (
    contract,
    emit,
    get_host,
    resolve_pot_id,
)
from adapters.inbound.cli.telemetry.onboarding_events import (
    capture_activation_succeeded,
    capture_onboarding_event,
)
from domain.errors import CapabilityNotImplemented

ingest_app = typer.Typer(help="Scanner ingestion + run history.")
dead_letter_app = typer.Typer(help="Dead-letter inspection + retry.")
ingest_app.add_typer(dead_letter_app, name="dead-letter")


def _needs_run_store(slot: str) -> None:
    # ingest history (show/replay/retry/dead-letter) is backed by the consumer
    # run store, which is the largest missing port (HU4). Until it lands these
    # are honest not-implemented seams.
    raise CapabilityNotImplemented(
        f"ingest.{slot}",
        detail="ingest run history is not persisted yet (consumer run store — HU4)",
        recommended_next_action="land the ledger run store (HU4) to back ingest history (HU6)",
    )


@ingest_app.command("scan")
def ingest_scan(
    path: str = typer.Option(".", "--path", help="Working-tree root to scan."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: active pot)."),
    repo_name: str = typer.Option(None, "--repo", help="Repo name to stamp on claims."),
    source: str = typer.Option(None, "--source"),  # reserved; per-source scoping TODO
    changed: bool = typer.Option(False, "--changed"),  # reserved; diff-only TODO
    watch: bool = typer.Option(False, "--watch"),  # reserved; watch mode TODO
) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        run_id = f"scan:{uuid.uuid4().hex}"
        result = host.ingest.scan_path(
            pot_id=pot_id, root=path, run_id=run_id, repo_name=repo_name
        )
        _capture_ingest_scan_activation(
            scanners_run=len(result.scanners_run),
            entities_upserted=result.entities_upserted,
            edges_upserted=result.edges_upserted,
        )
        emit(
            {
                "pot_id": pot_id,
                "run_id": run_id,
                "scanners_run": list(result.scanners_run),
                "entities_upserted": result.entities_upserted,
                "edges_upserted": result.edges_upserted,
                "skipped_files": result.skipped_files,
                "warnings": list(result.warnings),
            },
            human=(
                f"scanned {path}: {result.entities_upserted} entities, "
                f"{result.edges_upserted} claims "
                f"via [{', '.join(result.scanners_run) or 'no scanner matched'}]"
            ),
        )


@ingest_app.command("status")
def ingest_status() -> None:
    with contract():
        emit(
            {
                "runs": 0,
                "detail": "ingestion run history not persisted (local profile)",
            },
            human="ingestion: run history not persisted on the local profile",
        )


@ingest_app.command("runs")
def ingest_runs() -> None:
    with contract():
        emit({"runs": []}, human="(no persisted ingestion runs)")


@ingest_app.command("show")
def ingest_show(run_id: str) -> None:
    """Show one ingestion run (events, states, claims). [HU4/HU6]"""
    with contract():
        _needs_run_store("show")


@ingest_app.command("replay")
def ingest_replay(run_id: str) -> None:
    """Replay a run's events from the consumer run store. [HU4/HU6]"""
    with contract():
        _needs_run_store("replay")


@ingest_app.command("retry")
def ingest_retry(
    run_id: str,
    failed: bool = typer.Option(False, "--failed"),
    timed_out: bool = typer.Option(False, "--timed-out"),
) -> None:
    """Retry failed/timed-out events in a run. [HU4/HU6]"""
    with contract():
        _needs_run_store("retry")


@dead_letter_app.command("list")
def ingest_dead_letter_list() -> None:
    """List dead-lettered events. [HU4/HU6]"""
    with contract():
        _needs_run_store("dead_letter.list")


@dead_letter_app.command("retry")
def ingest_dead_letter_retry(event_id: str) -> None:
    """Re-enqueue a dead-lettered event. [HU4/HU6]"""
    with contract():
        _needs_run_store("dead_letter.retry")


__all__ = ["ingest_app"]


def _capture_ingest_scan_activation(
    *, scanners_run: int, entities_upserted: int, edges_upserted: int
) -> None:
    capture_onboarding_event(
        "cli_onboarding_ingest_scan_completed",
        phase="activation",
        entrypoint="direct_command",
        properties={
            "command": "ingest scan",
            "scanners_run_count": scanners_run,
            "entities_upserted": entities_upserted,
            "edges_upserted": edges_upserted,
        },
    )
    capture_activation_succeeded(
        command="ingest scan",
        result_kind="scan_result",
    )
