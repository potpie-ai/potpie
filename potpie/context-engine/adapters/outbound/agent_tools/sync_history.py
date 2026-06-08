"""Diff-sync history agent tools: ``read_sync_history`` / ``write_sync_history``.

These give the diff-sync playbooks a durable, append-only place to record each
graph-audit run and recover the previous cursor, backed by a
:class:`SyncHistoryStore`. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_sync_history_tools(...)])``;
the playbook ``tool_hints`` already allowlist both tool names so the agent's
server-side tool gate admits them only for the diff-sync event-kinds.

Both tools close over ``state.pot_id`` so history stays scoped to the pot being
reconciled — one pot can never read or append another pot's sync history.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from domain.error_redaction import safe_error
from domain.ports.sync_history import SyncHistoryStore

logger = logging.getLogger(__name__)


def _derive_key(record: dict[str, Any]) -> str | None:
    """Pull the scope key (team / project) out of a record."""
    for field in ("team", "project_key", "key"):
        value = record.get(field)
        if value:
            return str(value)
    return None


def build_sync_history_tools(
    store: SyncHistoryStore,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch builder exposing the diff-sync history tools.

    Args:
        store: Append-only history backend (e.g.
            :class:`FileSystemSyncHistoryStore`).

    Returns:
        A callable matching the agent's ``add_extra_tools`` contract.
    """

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; "
                    "skipping sync-history tools"
                )
                return []

        pot_id = getattr(state, "pot_id", None)

        def read_sync_history(
            source_system: str,
            scope: str,
            key: str,
            limit: int | None = 20,
        ) -> dict[str, Any]:
            """Read prior diff-sync audit records for one source scope.

            ``scope`` is the event_type (``linear_team`` / ``jira_project``);
            ``key`` is the team key / project key. Returns records oldest→newest
            so the last element holds the most recent ``new_cursor``.
            """
            try:
                records = store.read(
                    pot_id=pot_id,
                    source_system=source_system,
                    scope=scope,
                    key=key,
                    limit=limit,
                )
            except Exception as exc:
                logger.exception(
                    "read_sync_history failed pot=%s scope=%s key=%s",
                    pot_id,
                    scope,
                    key,
                )
                return {"error": safe_error(exc)}
            latest_cursor = records[-1].get("new_cursor") if records else None
            return {
                "count": len(records),
                "records": records,
                "latest_cursor": latest_cursor,
            }

        def write_sync_history(record: dict[str, Any]) -> dict[str, Any]:
            """Append one diff-sync audit record (append-only; never rewrites).

            ``source_system``, scope (``event_type``), and key (``team`` /
            ``project_key``) are read from the record the skill already built.
            """
            if not isinstance(record, dict):
                return {"error": "record must be an object"}
            src = record.get("source_system")
            scp = record.get("event_type")
            k = _derive_key(record)
            if not (src and scp and k):
                return {
                    "error": "missing_scope",
                    "message": (
                        "need source_system, scope (event_type), and key "
                        "(team/project_key) either as args or in the record"
                    ),
                }
            try:
                meta = store.append(
                    pot_id=pot_id,
                    source_system=str(src),
                    scope=str(scp),
                    key=str(k),
                    record=record,
                )
            except Exception as exc:
                logger.exception(
                    "write_sync_history failed pot=%s scope=%s key=%s",
                    pot_id,
                    scp,
                    k,
                )
                return {"error": safe_error(exc)}
            return {"ok": True, **meta}

        return [
            Tool(
                read_sync_history,
                name="read_sync_history",
                description=(
                    "Read the append-only diff-sync graph-audit history for one "
                    "source scope. Args: source_system (e.g. 'linear'|'jira'), "
                    "scope (the event_type, e.g. 'linear_team'|'jira_project'), "
                    "key (team key or project key), optional limit (most-recent "
                    "N, default 20). Returns records oldest->newest plus "
                    "latest_cursor — read this first to recover the previous "
                    "graph-audit cursor before enumerating source refs."
                ),
            ),
            Tool(
                write_sync_history,
                name="write_sync_history",
                description=(
                    "Append ONE diff-sync graph-audit record (JSON object) to "
                    "this scope's append-only history. source_system, scope "
                    "(event_type), and key (team/project_key) are read from the "
                    "record. Call exactly once per run, after graph writes, "
                    "with the final cursor/status/graph_checked/graph_missing/"
                    "graph_stale. Never use it to overwrite or compact prior lines."
                ),
            ),
        ]

    return _builder


__all__ = ["build_sync_history_tools"]
