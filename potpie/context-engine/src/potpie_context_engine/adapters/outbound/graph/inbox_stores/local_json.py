"""Local JSON graph inbox store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_core.graph_inbox import GraphInboxItem

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalJsonGraphInboxStore:
    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "graph_inbox.json"

    def save(self, item: GraphInboxItem) -> None:
        state = self._load()
        items = state.setdefault("items", {})
        by_pot = items.setdefault(item.pot_id, {})
        by_pot[item.item_id] = item.to_dict()
        self._save(state)

    def get(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        raw = self._load().get("items", {}).get(pot_id, {}).get(item_id)
        if not isinstance(raw, dict):
            return None
        return GraphInboxItem.from_dict(raw)

    def list(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphInboxItem, ...]:
        by_pot = self._load().get("items", {}).get(pot_id, {})
        if not isinstance(by_pot, dict):
            return ()
        records = [
            GraphInboxItem.from_dict(raw)
            for raw in by_pot.values()
            if isinstance(raw, dict)
        ]
        status_filter = {value for value in status if value}
        if status_filter:
            records = [record for record in records if record.status in status_filter]
        if claimed_by:
            records = [record for record in records if record.claimed_by == claimed_by]
        if suspected_subgraph:
            records = [
                record
                for record in records
                if suspected_subgraph in record.suspected_subgraphs
            ]
        if source_ref:
            records = [
                record
                for record in records
                if source_ref in record.source_refs or source_ref in record.evidence
            ]
        if since or until:
            records = [
                record
                for record in records
                if _record_in_window(record, since=since, until=until)
            ]
        records.sort(key=_sort_time, reverse=True)
        if limit is not None and limit >= 0:
            records = records[:limit]
        return tuple(records)

    def _load(self) -> dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"items": {}}
        except OSError as exc:
            logger.warning("graph inbox store unreadable at %s: %s", self._path, exc)
            return {"items": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("graph inbox store corrupt at %s; resetting", self._path)
            return {"items": {}}
        if not isinstance(data, dict):
            return {"items": {}}
        data.setdefault("items", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def _record_in_window(
    record: GraphInboxItem,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    for value in (record.created_at, record.claimed_at, record.closed_at):
        if value is None:
            continue
        if since is not None and value < since:
            continue
        if until is not None and value > until:
            continue
        return True
    return False


def _sort_time(record: GraphInboxItem) -> datetime:
    return record.closed_at or record.claimed_at or record.created_at


__all__ = ["LocalJsonGraphInboxStore"]
