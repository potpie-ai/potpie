"""Run one asynchronous runtime operation from a synchronous product surface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Execute ``operation`` without constructing it inside a running event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())
    raise RuntimeError("synchronous runtime surfaces cannot run inside an event loop")


__all__ = ["run_sync"]
