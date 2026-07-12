"""Explicit configuration for standalone context-engine construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EngineStorageMode = Literal["persistent", "in_memory"]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Library-only settings with no product path or environment discovery."""

    storage_mode: EngineStorageMode
    data_dir: Path | None
    backend: str
    profile: str = "local"

    def __post_init__(self) -> None:
        if self.storage_mode == "persistent" and self.data_dir is None:
            raise ValueError("persistent engine configuration requires data_dir")
        if self.storage_mode == "in_memory" and self.data_dir is not None:
            raise ValueError("in-memory engine configuration cannot define data_dir")
        if not self.backend.strip():
            raise ValueError("engine backend cannot be empty")

    @classmethod
    def persistent(
        cls,
        *,
        data_dir: str | Path,
        backend: str = "embedded",
        profile: str = "local",
    ) -> EngineConfig:
        path = Path(data_dir).expanduser()
        return cls(
            storage_mode="persistent",
            data_dir=path,
            backend=backend,
            profile=profile,
        )

    @classmethod
    def in_memory(cls, *, profile: str = "local") -> EngineConfig:
        return cls(
            storage_mode="in_memory",
            data_dir=None,
            backend="in_memory",
            profile=profile,
        )


__all__ = ["EngineConfig", "EngineStorageMode"]
