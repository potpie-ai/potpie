"""Locking port for workspace orchestration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol


class LockManager(Protocol):
    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        ...

