"""Local JSON graph-plan store.

This is the first local workbench implementation. It persists under the Potpie
home so proposed plans survive separate CLI invocations; hosted installs can
swap in a transactional store behind the same port.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_core.graph_plans import GraphMutationPlanRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalJsonGraphPlanStore:
    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "graph_plans.json"

    def save(self, record: GraphMutationPlanRecord) -> None:
        state = self._load()
        plans = state.setdefault("plans", {})
        by_pot = plans.setdefault(record.pot_id, {})
        by_pot[record.plan_id] = record.to_dict()
        self._save(state)

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        raw = self._load().get("plans", {}).get(pot_id, {}).get(plan_id)
        if not isinstance(raw, dict):
            return None
        return GraphMutationPlanRecord.from_dict(raw)

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        by_pot = self._load().get("plans", {}).get(pot_id, {})
        if not isinstance(by_pot, dict):
            return ()
        records = [
            GraphMutationPlanRecord.from_dict(raw)
            for raw in by_pot.values()
            if isinstance(raw, dict)
        ]
        if plan_id:
            records = [record for record in records if record.plan_id == plan_id]
        if mutation_id:
            records = [
                record for record in records if record.mutation_id == mutation_id
            ]
        if since or until:
            records = [
                record
                for record in records
                if _record_in_window(record, since=since, until=until)
            ]
        records.sort(
            key=lambda record: record.committed_at or record.created_at,
            reverse=True,
        )
        if limit is not None and limit >= 0:
            records = records[:limit]
        return tuple(records)

    def _load(self) -> dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"plans": {}}
        except OSError as exc:
            logger.warning("graph plan store unreadable at %s: %s", self._path, exc)
            return {"plans": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("graph plan store corrupt at %s; resetting", self._path)
            return {"plans": {}}
        if not isinstance(data, dict):
            return {"plans": {}}
        data.setdefault("plans", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def _record_in_window(
    record: GraphMutationPlanRecord,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    times = (record.created_at, record.committed_at)
    for value in times:
        if value is None:
            continue
        if since is not None and value < since:
            continue
        if until is not None and value > until:
            continue
        return True
    return False


__all__ = ["LocalJsonGraphPlanStore"]
