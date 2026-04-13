from __future__ import annotations

import json
from typing import Any

from seed.paths import state_path


class SeedState(dict[str, Any]):
    """Persisted seed state (ids per entity type)."""

    @classmethod
    def load(cls, profile: str) -> "SeedState | None":
        path = state_path(profile)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save(self, profile: str) -> None:
        path = state_path(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self), f, indent=2)

    def remove_file(self, profile: str) -> None:
        path = state_path(profile)
        if path.exists():
            path.unlink()
