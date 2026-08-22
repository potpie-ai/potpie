"""Narrow operation coordination derived from typed safety metadata."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from potpie.runtime.operations import OperationSpec, SafetyClass
from potpie_context_engine import ContextIdentity
from potpie_context_engine.requests import EngineRequest


LockMode = Literal["read", "write"]


@dataclass(frozen=True, order=True, slots=True)
class ConflictKey:
    kind: Literal["context", "resource", "daemon"]
    identity: str


class _ReadWriteLock:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0
            )
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0
                )
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


class OperationCoordinator:
    """Coordinate only operations whose typed conflict keys overlap."""

    def __init__(self) -> None:
        self._locks: dict[ConflictKey, _ReadWriteLock] = {}
        self._locks_guard = asyncio.Lock()

    @asynccontextmanager
    async def coordinate(
        self,
        *,
        spec: OperationSpec,
        context: ContextIdentity,
        request: EngineRequest,
    ) -> AsyncIterator[None]:
        requirements = self._requirements(spec, context, request)
        async with AsyncExitStack() as stack:
            for key, mode in requirements:
                lock = await self._lock_for(key)
                if mode == "read":
                    await stack.enter_async_context(lock.read())
                else:
                    await stack.enter_async_context(lock.write())
            yield

    @asynccontextmanager
    async def lifecycle_control(self) -> AsyncIterator[None]:
        lock = await self._lock_for(ConflictKey("daemon", "process"))
        async with lock.write():
            yield

    async def _lock_for(self, key: ConflictKey) -> _ReadWriteLock:
        async with self._locks_guard:
            return self._locks.setdefault(key, _ReadWriteLock())

    @staticmethod
    def _requirements(
        spec: OperationSpec,
        context: ContextIdentity,
        request: EngineRequest,
    ) -> tuple[tuple[ConflictKey, LockMode], ...]:
        context_key = ConflictKey("context", context.value)
        if spec.safety is SafetyClass.SHARED_CONTEXT_READ:
            return ((context_key, "read"),)
        if spec.safety is SafetyClass.EXCLUSIVE_CONTEXT_MUTATION:
            return ((context_key, "write"),)
        if spec.safety is SafetyClass.EXCLUSIVE_RESOURCE_MUTATION:
            resource_identity = ":".join(
                str(getattr(request, field)) for field in spec.resource_identity_fields
            )
            resource_key = ConflictKey(
                "resource", f"{spec.resource_type}:{resource_identity}"
            )
            return tuple(sorted(((context_key, "write"), (resource_key, "write"))))
        raise ValueError(
            f"{spec.operation.value} is not a context or resource operation"
        )


__all__ = ["ConflictKey", "OperationCoordinator"]
