"""``LocalConfigService`` — local home dir + JSON config file.

Backs the first setup step. State lives at ``<home>/config.json`` where
``<home>`` is ``$CONTEXT_ENGINE_HOME`` or ``~/.potpie`` (shared with
:func:`adapters.outbound.pots.local_pot_store.default_home`). This is a working
Real dirs + JSON, not a stub — config is cheap and unblocks every
downstream step. The real config layer may add schema/validation behind the same
``ConfigService`` interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_engine.adapters.outbound.pots.local_pot_store import default_home
from context_engine.domain.lifecycle import SetupPlan


@dataclass(slots=True)
class LocalConfigService:
    """Flat-file config provisioning + get/set."""

    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "config.json"

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.home

    def write_defaults(self, plan: SetupPlan) -> Path:
        self.ensure_home()
        data = self._load()
        # Only fill values the user has not already set (idempotent re-runs).
        data.setdefault("profile", plan.mode)
        data.setdefault("backend", plan.backend)
        data.setdefault("home", str(self.home))
        self._save(data)
        return self._path

    def get(self, key: str) -> str | None:
        value = self._load().get(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self.ensure_home()
        self._save(data)

    def probe(self) -> dict[str, Any]:
        return {"home": str(self.home), "config_exists": self._path.exists()}

    # --- raw state ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self._path)


__all__ = ["LocalConfigService"]
