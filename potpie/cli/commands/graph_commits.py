"""Graph commit history, journal status and explicit preview-then-apply rollback.

Every command is one typed operation. Reads (``journal-status``, ``commits``,
``commit-show``) list what the native journal recorded. ``revert`` and
``rollback`` only ever build a server-held preview: nothing in the graph
changes until ``apply-preview`` names that preview, after explicit destructive
confirmation, and the server re-checks HEAD, the resource generation and the
inverse before writing.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping
from typing import Any

import typer

from potpie_context_engine.requests import (
    ApplyPreviewRequest,
    CommitShowRequest,
    CommitsRequest,
    DisableRollbackRequest,
    JournalStatusRequest,
    RebuildCommitsRequest,
    RevertPreviewRequest,
    RollbackPreviewRequest,
)

_PREVIEW_ONLY = "Use --preview, then graph apply-preview <preview-id>."


def apply_preview_command(preview_id: str, pot_id: str) -> str:
    """The exact command that applies a preview, for humans and agents alike."""

    return (
        f"potpie graph apply-preview {shlex.quote(preview_id)} "
        f"--pot {shlex.quote(pot_id)} --yes"
    )


def _run(
    command: str,
    pot: str | None,
    call: Callable[[Any], Any],
    *,
    destructive: Callable[[str], Any] | None = None,
) -> None:
    from potpie.cli.commands.graph import (
        _emit_graph_result,
        _graph_command,
        get_engine_client,
        get_root_runtime,
        resolve_pot_id,
        run_engine_operation,
    )

    with _graph_command(command) as ctx:
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        ctx.set_pot_id(pot_id)
        if destructive is None:
            result = run_engine_operation(call(get_engine_client(pot)))
        else:
            # Destructive intent binds the exact context, so name it by id.
            confirmation = destructive(pot_id)
            result = run_engine_operation(
                call(get_engine_client(pot_id), confirmation=confirmation)
            )
        payload = _with_reason_message(result.to_dict())
        next_action = None
        preview = payload.get("preview")
        if isinstance(preview, Mapping) and preview.get("preview_id"):
            next_action = apply_preview_command(str(preview["preview_id"]), pot_id)
        _emit_graph_result(
            ctx,
            payload,
            human=_human(payload, pot_id=pot_id),
            recommended_next_action=next_action,
        )


def _with_reason_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Lift the first refusal reason into ``message`` for the error envelope."""

    if payload.get("ok", True) is not False or payload.get("message"):
        return payload
    reasons = payload.get("reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
        message = reasons[0].get("message")
        if message:
            return {**payload, "message": str(message)}
    return payload


def _human(payload: Mapping[str, Any], *, pot_id: str) -> str:
    lines: list[str] = []
    for row in payload.get("headers", ()) or ():
        coverage = (
            "revertible"
            if row.get("rollback_supported")
            else row.get("unsupported_reason") or "barrier"
        )
        lines.append(
            f"#{row.get('sequence')} {row.get('commit_id')} · {row.get('message')} · "
            f"{row.get('actor')} · {row.get('affected_record_count')} records · "
            f"{coverage}"
        )
    preview = payload.get("preview")
    if isinstance(preview, Mapping):
        lines.append(
            f"Preview {preview.get('preview_id')}: "
            f"{payload.get('affected_record_count')} records; requires "
            f"{preview.get('required_access')}; expires {preview.get('expires_at')}"
        )
        lines.append(
            "Apply with: "
            + apply_preview_command(str(preview.get("preview_id")), pot_id)
        )
    commit = payload.get("commit")
    if isinstance(commit, Mapping):
        replayed = " (already applied)" if payload.get("replayed") else ""
        lines.append(
            f"Committed #{commit.get('sequence')} {commit.get('commit_id')}{replayed}"
        )
    coverage = payload.get("coverage")
    if isinstance(coverage, Mapping):
        if coverage.get("legacy_only"):
            lines.append(
                "No journal coverage. Earlier changes remain in graph history."
            )
        elif coverage:
            lines.append(
                f"HEAD {coverage.get('head')}; indexing lag {coverage.get('indexing_lag')}"
            )
    if payload.get("next_cursor"):
        lines.append(f"Next cursor: {payload['next_cursor']}")
    if lines:
        return "\n".join(lines)
    import json

    return json.dumps(payload, indent=2, default=str)


def _confirm_apply(preview_id: str, *, yes: bool) -> Callable[[str], Any]:
    from potpie.cli.commands._common import confirm_destructive_operation

    def confirm(pot_id: str) -> Any:
        return confirm_destructive_operation(
            confirmed_by_flag=yes,
            prompt=(
                f"Apply rollback preview '{preview_id}' to context '{pot_id}'? "
                "This changes graph data."
            ),
            rerun_command=apply_preview_command(preview_id, pot_id),
        )

    return confirm


def register_commit_commands(app: typer.Typer) -> None:
    @app.command("journal-status")
    def journal_status(pot: str | None = typer.Option(None, "--pot")) -> None:
        """Inspect native coverage, capability, resource guard, and atomic limits."""
        _run(
            "graph.journal-status",
            pot,
            lambda client: client.journal_status(JournalStatusRequest()),
        )

    @app.command("disable-rollback")
    def disable_rollback(pot: str | None = typer.Option(None, "--pot")) -> None:
        """Disable rollback while retaining journal capture (admin)."""
        _run(
            "graph.disable-rollback",
            pot,
            lambda client: client.disable_rollback(DisableRollbackRequest()),
        )

    @app.command("rebuild-commits")
    def rebuild_commits(pot: str | None = typer.Option(None, "--pot")) -> None:
        """Rebuild the selected pot's derived listing index (admin)."""
        _run(
            "graph.rebuild-commits",
            pot,
            lambda client: client.rebuild_commits(RebuildCommitsRequest()),
        )

    @app.command("commits")
    def commits(
        pot: str | None = typer.Option(None, "--pot"),
        cursor: str | None = typer.Option(None, "--cursor"),
        limit: int = typer.Option(50, "--limit", min=1, max=200),
        actor: str | None = typer.Option(None, "--actor"),
        origin: str | None = typer.Option(None, "--origin"),
        logical_key: str | None = typer.Option(None, "--entity"),
    ) -> None:
        """List actual mutation commits with bounded keyset pagination."""
        _run(
            "graph.commits",
            pot,
            lambda client: client.commits(
                CommitsRequest(
                    cursor=cursor,
                    limit=limit,
                    actor=actor,
                    origin=origin,
                    logical_key=logical_key,
                )
            ),
        )

    @app.command("commit-show")
    def commit_show(
        commit_id: str,
        pot: str | None = typer.Option(None, "--pot"),
        offset: int = typer.Option(0, "--offset", min=0),
        limit: int = typer.Option(100, "--limit", min=1, max=200),
    ) -> None:
        """Show recorded changes, with partial historical context."""
        _run(
            "graph.commit-show",
            pot,
            lambda client: client.commit_show(
                CommitShowRequest(commit_id=commit_id, offset=offset, limit=limit)
            ),
        )

    @app.command("revert")
    def revert(
        commit_id: str,
        expected_head: str = typer.Option(..., "--expected-head"),
        preview: bool = typer.Option(False, "--preview"),
        pot: str | None = typer.Option(None, "--pot"),
    ) -> None:
        """Preview selective revert; apply its server preview with apply-preview."""
        if not preview:
            raise typer.BadParameter(_PREVIEW_ONLY)
        _run(
            "graph.revert",
            pot,
            lambda client: client.revert_preview(
                RevertPreviewRequest(commit_id=commit_id, expected_head=expected_head)
            ),
        )

    @app.command("rollback")
    def rollback(
        target: str = typer.Option(..., "--to"),
        expected_head: str = typer.Option(..., "--expected-head"),
        preview: bool = typer.Option(False, "--preview"),
        pot: str | None = typer.Option(None, "--pot"),
    ) -> None:
        """Preview an atomic rollback of every commit after the target."""
        if not preview:
            raise typer.BadParameter(_PREVIEW_ONLY)
        _run(
            "graph.rollback",
            pot,
            lambda client: client.rollback_preview(
                RollbackPreviewRequest(
                    target_commit_id=target, expected_head=expected_head
                )
            ),
        )

    @app.command("apply-preview")
    def apply_preview(
        preview_id: str,
        pot: str | None = typer.Option(None, "--pot"),
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Confirm applying the previewed change to the graph.",
        ),
    ) -> None:
        """Apply a persisted preview after renewed permission and native guard checks."""
        _run(
            "graph.apply-preview",
            pot,
            lambda client, confirmation: client.apply_preview(
                ApplyPreviewRequest(preview_id=preview_id),
                confirmation=confirmation,
            ),
            destructive=_confirm_apply(preview_id, yes=yes),
        )


__all__ = ["apply_preview_command", "register_commit_commands"]
