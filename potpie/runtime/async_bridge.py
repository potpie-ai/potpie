"""Persistent async runner for synchronous product surfaces."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRuntimeRunner:
    """Own one background event loop and submit coroutine factories to it."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def run(self, operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
        loop = self._ensure_started()
        result: Future[T] = Future()

        def start_operation() -> None:
            try:
                coroutine = operation()
                task = loop.create_task(coroutine)
            except BaseException as exc:
                result.set_exception(exc)
                return

            def complete(completed: asyncio.Task[T]) -> None:
                if completed.cancelled():
                    result.cancel()
                    return
                exception = completed.exception()
                if exception is not None:
                    result.set_exception(exception)
                    return
                result.set_result(completed.result())

            task.add_done_callback(complete)

        loop.call_soon_threadsafe(start_operation)
        return result.result()

    def shutdown(self) -> None:
        with self._state_lock:
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                return
            loop.call_soon_threadsafe(loop.stop)
        thread.join()
        with self._state_lock:
            if self._thread is thread:
                self._loop = None
                self._thread = None
                self._ready.clear()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._state_lock:
            if self._thread is None:
                self._ready.clear()
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name="potpie-runtime-loop",
                    daemon=True,
                )
                self._thread.start()
        self._ready.wait()
        loop = self._loop
        if loop is None:
            raise RuntimeError("Potpie runtime event loop failed to start")
        return loop

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()


_runner = AsyncRuntimeRunner()


def run_sync(operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async operation on the product runtime loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _runner.run(operation)
    raise RuntimeError("synchronous runtime surfaces cannot run inside an event loop")


def shutdown_async_bridge() -> None:
    """Stop the product runtime loop after all owned resources are closed."""

    _runner.shutdown()


__all__ = ["AsyncRuntimeRunner", "run_sync", "shutdown_async_bridge"]
