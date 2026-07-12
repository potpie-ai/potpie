"""Persistent product settings independent from engine configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KNOWN_CONFIG_KEYS: tuple[str, ...] = (
    "runtime_mode",
    "backend",
    "ledger.binding",
    "ledger.org",
    "ledger.url",
)

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATOR_RE = re.compile(r"[_\-.\s]+")
_SINGLE_WORD_SECRET_MARKERS = frozenset({"token", "secret", "password", "credential"})
_COMPOUND_SECRET_MARKERS = ("apikey",)


@dataclass(slots=True)
class ProductConfigService:
    data_dir: Path

    @property
    def path(self) -> Path:
        return self.data_dir / "config.json"

    def get(self, key: str) -> Any:
        return self._load().get(key)

    def set(self, key: str, value: Any) -> None:
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("configuration key cannot be empty")
        data = self._load()
        data[clean_key] = value
        self._save(data)

    def list(self) -> dict[str, Any]:
        return self._load()

    def list_public(self) -> dict[str, Any]:
        return {
            key: "<redacted>" if _secret_key(key) else value
            for key, value in self._load().items()
        }

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.path)


def _secret_key(key: str) -> bool:
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", _SEPARATOR_RE.sub(" ", key))
    words = [word for word in spaced.lower().split() if word]
    if any(word in _SINGLE_WORD_SECRET_MARKERS for word in words):
        return True
    joined = "".join(words)
    return any(marker in joined for marker in _COMPOUND_SECRET_MARKERS)


def is_secret_config_key(key: str) -> bool:
    return _secret_key(key)


def public_config_value(key: str, value: Any) -> str | None:
    if value is None:
        return None
    return "<redacted>" if _secret_key(key) else str(value)


__all__ = [
    "KNOWN_CONFIG_KEYS",
    "ProductConfigService",
    "is_secret_config_key",
    "public_config_value",
]
