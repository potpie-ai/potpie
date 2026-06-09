"""Port for diff-sync graph-audit history (append-only JSONL per source scope).

The diff-sync skills (``linear_team_diff_sync`` / ``jira_project_diff_sync``)
need durable, append-only audit records so each run can recover the previous
graph-audit cursor and never re-walk source history it already reconciled. This
port is the narrow surface the ``read_sync_history`` / ``write_sync_history``
agent tools depend on, so hosts can back it with the local filesystem, object
storage, or a managed run-history service without the tools changing.

Records are opaque JSON objects shaped by the skill (see the skill markdown for
the field list). The store treats them as append-only audit data: it never
rewrites or compacts existing lines.
"""

from __future__ import annotations

from typing import Any, Protocol


class SyncHistoryStore(Protocol):
    """Append-only history of diff-sync graph audits, scoped per source ref.

    Scoping is ``(pot_id, source_system, scope, key)`` where ``scope`` is the
    event_type (e.g. ``linear_team`` / ``jira_project``) and ``key`` is the
    team key / project key. Implementations derive a stable storage location
    from that tuple so repeated runs for the same team/project append to one
    history.
    """

    def read(
        self,
        *,
        pot_id: str | None,
        source_system: str,
        scope: str,
        key: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return prior audit records oldest→newest (empty when none exist).

        ``limit`` returns only the most recent ``limit`` records (still in
        chronological order) so a run can cheaply read the latest cursor.
        """
        ...

    def append(
        self,
        *,
        pot_id: str | None,
        source_system: str,
        scope: str,
        key: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one audit record. Returns ``{path, written}`` metadata."""
        ...


__all__ = ["SyncHistoryStore"]
