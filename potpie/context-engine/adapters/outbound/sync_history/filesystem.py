"""Local-filesystem implementation of :class:`SyncHistoryStore`.

Stores one append-only JSONL file per ``(pot, source scope, key)``, matching
the path the diff-sync skills document:

    <base>/<pot-id>/context-sync-history/<scope-slug>-<key-slug>.jsonl

e.g. ``<base>/pot-7/context-sync-history/linear-team-eng.jsonl`` and
``<base>/pot-7/context-sync-history/jira-project-proj.jsonl``. The base dir
comes from ``CONTEXT_ENGINE_SYNC_HISTORY_DIR`` (falling back to
``./.context-sync-history``) so operators can point it at a mounted volume.

Append-only by contract: :meth:`append` only ever opens files in ``"a"`` mode
and :meth:`read` never mutates, so historical audit lines are immutable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """Lowercase, collapse non-alphanumerics to single hyphens, trim."""
    return _SLUG_RE.sub("-", str(value).strip().lower()).strip("-") or "unknown"


class FileSystemSyncHistoryStore:
    """JSONL-per-scope diff-sync history rooted at ``base_dir``."""

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self._base = Path(base_dir)

    def _path(
        self, pot_id: str | None, scope: str, key: str
    ) -> Path:
        # ``scope`` is the event_type (linear_team / jira_project); the file
        # name mirrors the skill's documented pattern (underscores → hyphens).
        filename = f"{_slug(scope.replace('_', '-'))}-{_slug(key)}.jsonl"
        return (
            self._base
            / _slug(pot_id or "_")
            / "context-sync-history"
            / filename
        )

    def read(
        self,
        *,
        pot_id: str | None,
        source_system: str,
        scope: str,
        key: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        path = self._path(pot_id, scope, key)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    # A corrupt line must not crash an audit — skip and warn.
                    logger.warning("skipping malformed sync-history line in %s", path)
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
        if limit is not None and limit >= 0:
            return records[-limit:]
        return records

    def append(
        self,
        *,
        pot_id: str | None,
        source_system: str,
        scope: str,
        key: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._path(pot_id, scope, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return {"path": str(path), "written": True}


__all__ = ["FileSystemSyncHistoryStore"]
