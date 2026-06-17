"""ShellContext: cross-cutting services handed to every plugin instance."""

from __future__ import annotations

import asyncio
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any


class ServiceEndpoints:
    """Registry of resolved managed-service endpoints."""

    def __init__(self) -> None:
        self._eps: dict[str, str] = {}

    def set(self, name: str, endpoint: str) -> None:
        self._eps[name] = endpoint

    def get(self, name: str) -> str:
        return self._eps[name]

    def remove(self, name: str) -> None:
        self._eps.pop(name, None)

    def resolve(self, value: str) -> str:
        """``service:<name>`` resolves to the endpoint; other values pass through."""
        if value.startswith("service:"):
            name = value.split(":", 1)[1]
            try:
                return self._eps[name]
            except KeyError as exc:
                raise KeyError(f"unknown service endpoint: {name!r}") from exc
        return value


@dataclass
class ShellContext:
    config: dict[str, Any]
    data_dir: pathlib.Path
    logger: logging.Logger
    endpoints: ServiceEndpoints
    shutdown: asyncio.Event = field(default_factory=asyncio.Event)
