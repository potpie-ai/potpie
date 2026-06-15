"""Local JSON graph-plan store.

This is the first local workbench implementation. It persists under the Potpie
home so proposed plans survive separate CLI invocations; hosted installs can
swap in a transactional store behind the same port.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.outbound.pots.local_pot_store import default_home
from domain.graph_plans import GraphMutationPlanRecord

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


__all__ = ["LocalJsonGraphPlanStore"]
