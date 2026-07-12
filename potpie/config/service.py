"""Persistent product settings independent from engine configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    normalized = key.lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in ("token", "secret", "password", "api_key", "credential")
    )


__all__ = ["ProductConfigService"]
