"""Commit history and explicit preview/apply commands."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime

import typer


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _run(command, pot, method, *args, **kwargs):
    from potpie.cli.commands.graph import (
        _emit_graph_result,
        _graph_command,
        get_host,
        resolve_pot_id,
    )

    with _graph_command(command) as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = getattr(host.graph_workbench, method)(*args, pot_id=pot_id, **kwargs)
        payload = json.loads(json.dumps(result, default=_json_default, allow_nan=False))
        lines = []
        for row in payload.get("headers", ()):
            coverage = (
                "revertible"
                if row["rollback_supported"]
                else row.get("unsupported_reason") or "barrier"
            )
            lines.append(
                f"#{row['sequence']} {row['commit_id']} · {row['message']} · {row['actor']} · {row['affected_record_count']} records · {coverage}"
            )
        if "preview" in payload:
            preview = payload["preview"]
            lines.append(
                f"Preview {preview['preview_id']}: {payload['affected_record_count']} records; requires {preview['required_access']}; expires {preview['expires_at']}"
            )
            lines.append(
                f"Apply with: potpie graph apply-preview {preview['preview_id']} --pot {pot or pot_id}"
            )
        coverage = payload.get("coverage", {})
        if coverage.get("legacy_only"):
            lines.append(
                "No journal coverage. Earlier changes remain in graph history."
            )
        elif coverage:
            lines.append(
                f"HEAD {coverage['head']}; indexing lag {coverage['indexing_lag']}"
            )
        if payload.get("next_cursor"):
            lines.append(f"Next cursor: {payload['next_cursor']}")
        _emit_graph_result(
            ctx, payload, human="\n".join(lines) or json.dumps(payload, indent=2)
        )


def register_commit_commands(app: typer.Typer):
    @app.command("journal-status")
    def journal_status(pot: str | None = typer.Option(None, "--pot")):
        """Inspect native coverage, capability, resource guard, and atomic limits."""
        _run("graph.journal-status", pot, "journal_status")

    @app.command("disable-rollback")
    def disable_rollback(pot: str | None = typer.Option(None, "--pot")):
        """Disable rollback while retaining journal capture (admin)."""
        _run("graph.disable-rollback", pot, "disable_rollback")

    @app.command("rebuild-commits")
    def rebuild_commits(pot: str | None = typer.Option(None, "--pot")):
        """Rebuild the selected pot's derived listing index (admin)."""
        _run("graph.rebuild-commits", pot, "rebuild_commits")

    @app.command("commits")
    def commits(
        pot: str | None = typer.Option(None, "--pot"),
        cursor: str | None = typer.Option(None, "--cursor"),
        limit: int = typer.Option(50, "--limit", min=1, max=200),
        actor: str | None = typer.Option(None, "--actor"),
        origin: str | None = typer.Option(None, "--origin"),
        logical_key: str | None = typer.Option(None, "--entity"),
    ):
        """List actual mutation commits with bounded keyset pagination."""
        _run(
            "graph.commits",
            pot,
            "commits",
            cursor=cursor,
            limit=limit,
            actor=actor,
            origin=origin,
            logical_key=logical_key,
        )

    @app.command("commit-show")
    def commit_show(
        commit_id: str,
        pot: str | None = typer.Option(None, "--pot"),
        offset: int = typer.Option(0, "--offset", min=0),
        limit: int = typer.Option(100, "--limit", min=1, max=200),
    ):
        """Show recorded changes, with partial historical context."""
        _run(
            "graph.commit-show",
            pot,
            "commit_show",
            commit_id,
            offset=offset,
            limit=limit,
        )

    @app.command("revert")
    def revert(
        commit_id: str,
        expected_head: str = typer.Option(..., "--expected-head"),
        preview: bool = typer.Option(False, "--preview"),
        pot: str | None = typer.Option(None, "--pot"),
    ):
        """Preview selective revert; apply its server preview with apply-preview."""
        if not preview:
            raise typer.BadParameter(
                "Use --preview, then graph apply-preview <preview-id>."
            )
        _run(
            "graph.revert",
            pot,
            "revert_preview",
            commit_id,
            expected_head=expected_head,
        )

    @app.command("rollback")
    def rollback(
        target: str = typer.Option(..., "--to"),
        expected_head: str = typer.Option(..., "--expected-head"),
        preview: bool = typer.Option(False, "--preview"),
        pot: str | None = typer.Option(None, "--pot"),
    ):
        """Preview an atomic rollback of every commit after the target."""
        if not preview:
            raise typer.BadParameter(
                "Use --preview, then graph apply-preview <preview-id>."
            )
        _run(
            "graph.rollback",
            pot,
            "rollback_preview",
            target,
            expected_head=expected_head,
        )

    @app.command("apply-preview")
    def apply_preview(preview_id: str, pot: str | None = typer.Option(None, "--pot")):
        """Apply a persisted preview after renewed permission and native guard checks."""
        _run("graph.apply-preview", pot, "apply_preview", preview_id)
