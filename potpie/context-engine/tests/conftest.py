"""Shared pytest fixtures for the context-engine test suite."""

from __future__ import annotations

import logging
import shutil
import socket
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from potpie_context_engine.host.daemon_runtime.context import ServiceEndpoints, ShellContext


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def short_socket_dir() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp(prefix="potpie-d-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_for_condition(
    condition: Callable[[], bool],
    *,
    timeout_s: float = 2.5,
    interval_s: float = 0.05,
    error_message: str = "condition was not met before timeout",
) -> None:
    import asyncio

    remaining = timeout_s
    while remaining > 0:
        if condition():
            return
        await asyncio.sleep(interval_s)
        remaining -= interval_s
    raise TimeoutError(error_message)


@pytest.fixture()
def daemon_ctx(tmp_path: Path) -> ShellContext:
    return ShellContext(
        config={},
        data_dir=tmp_path,
        logger=logging.getLogger("test"),
        endpoints=ServiceEndpoints(),
    )


@pytest.fixture(autouse=True)
def _default_in_process_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep engine tests deterministic unless they explicitly select daemon mode."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
