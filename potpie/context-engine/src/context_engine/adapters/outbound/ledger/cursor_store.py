"""Local JSON-file ledger cursor store (POC).

Cursors live with the consumer graph, keyed by (pot, source), so the same
ledger can feed multiple graphs at independent positions.

    TODO(stage-N): move into the local state DB alongside pot state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from context_engine.adapters.outbound.pots.local_pot_store import default_home
from context_engine.domain.ports.ledger.client import LedgerCursor


@dataclass(slots=True)
class LocalLedgerCursorStore:
    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "ledger_cursors.json"

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, str]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def get(self, *, pot_id: str, source_id: str) -> LedgerCursor | None:
        token = self._load().get(f"{pot_id}:{source_id}")
        return (
            LedgerCursor(source_id=source_id, token=token)
            if token is not None
            else None
        )

    def set(self, *, pot_id: str, cursor: LedgerCursor) -> None:
        data = self._load()
        data[f"{pot_id}:{cursor.source_id}"] = cursor.token or ""
        self._save(data)


__all__ = ["LocalLedgerCursorStore"]
