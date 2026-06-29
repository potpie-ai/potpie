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

from adapters.outbound.pots.local_pot_store import default_home
from domain.lifecycle import SetupPlan

KNOWN_CONFIG_KEYS: tuple[str, ...] = (
    "profile",
    "backend",
    "home",
    "ledger.binding",
    "ledger.org",
    "ledger.url",
)

_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "api_key",
    "api-key",
    "credential",
)

_REDACTED = "<redacted>"


def is_secret_config_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def public_config_value(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if is_secret_config_key(key):
        return _REDACTED
    return str(value)


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

    def list_public(self) -> dict[str, str | None]:
        """Return all config entries with secret-like keys redacted."""
        return {
            key: public_config_value(key, value)
            for key, value in sorted(self._load().items())
        }

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


__all__ = [
    "KNOWN_CONFIG_KEYS",
    "LocalConfigService",
    "is_secret_config_key",
    "public_config_value",
]
