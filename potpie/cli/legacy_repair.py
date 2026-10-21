"""CLI preview/apply flow for a persisted legacy graph repair."""

from __future__ import annotations

import json
import shlex
import typer

from potpie_context_engine.requests import RepairRequest
from potpie.cli.commands._common import (
    confirm_destructive_operation,
    get_engine_client,
    run_engine_operation,
)


def run_legacy_repair(
    *, pot_id: str, dry_run: bool, apply: bool, repair_id: str | None, yes: bool
):
    if dry_run == apply:
        raise typer.BadParameter(
            "Choose exactly one of --dry-run or --apply with --legacy."
        )
    if apply and not repair_id:
        raise typer.BadParameter(
            "--apply requires the --repair-id returned by --dry-run."
        )
    if dry_run and yes:
        raise typer.BadParameter("--yes is only used to confirm --apply.")
    confirmation = None
    if apply:
        confirmation = confirm_destructive_operation(
            confirmed_by_flag=yes,
            prompt=f"Apply reviewed legacy repair '{repair_id}' in context '{pot_id}'?",
            rerun_command=shlex.join(
                [
                    "potpie",
                    "graph",
                    "repair",
                    "--legacy",
                    "--apply",
                    "--repair-id",
                    repair_id,
                    "--pot",
                    pot_id,
                    "--yes",
                ]
            ),
        )
    return run_engine_operation(
        get_engine_client(pot_id).repair(
            RepairRequest(
                legacy_mode="dry_run" if dry_run else "apply", repair_id=repair_id
            ),
            confirmation=confirmation,
        )
    )


def format_legacy_repair(report, *, limit: int = 20) -> str:
    """Show bounded review content; JSON output always contains the complete run."""
    lines = [
        f"{report.status}: repair {report.repair_id}; "
        f"{report.repairable_records} records, {report.completed_batches}/{report.total_batches} batches.",
        report.detail or "",
    ]
    for change in report.changes[:limit]:
        lines.append(
            f"{change['kind']} {change['key'] or '(no logical key)'} [{change['before_fingerprint']}]"
        )
        for namespace in ("field_updates", "property_updates"):
            if change[namespace]:
                lines.append(
                    f"  {namespace}: {json.dumps(change[namespace], sort_keys=True, ensure_ascii=True)}"
                )
    for finding in report.findings[:limit]:
        lines.append(f"{finding.target}: {finding.detail}")
    if len(report.changes) > limit or len(report.findings) > limit:
        lines.append(
            f"Display limited to {limit} changes and {limit} findings. Use --json to review the full frozen repair."
        )
    return "\n".join(line for line in lines if line)
