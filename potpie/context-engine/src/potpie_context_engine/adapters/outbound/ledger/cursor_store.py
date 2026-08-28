"""Local JSON-file ledger cursor store (POC).

Cursors live with the consumer graph, keyed by (pot, source), so the same
ledger can feed multiple graphs at independent positions.

    TODO(stage-N): move into the local state DB alongside pot state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from potpie_context_engine.adapters.outbound.graph._local_json_atomic import (
    locked_json_store,
)
from potpie_context_engine.adapters.outbound.local_paths import default_home
from potpie_context_engine.domain.ports.ledger.client import LedgerCursor


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
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self._path)

    def get(self, *, pot_id: str, source_id: str) -> LedgerCursor | None:
        token = self._load().get(f"{pot_id}:{source_id}")
        return (
            LedgerCursor(source_id=source_id, token=token)
            if token is not None
            else None
        )

    def set(self, *, pot_id: str, cursor: LedgerCursor) -> None:
        with locked_json_store(self._path):
            data = self._load()
            data[f"{pot_id}:{cursor.source_id}"] = cursor.token or ""
            self._save(data)


__all__ = ["LocalLedgerCursorStore"]
