"""Shared pytest fixtures for the context-engine test suite."""

from __future__ import annotations

import shutil
import socket
import sys
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if (_REPO_ROOT / "potpie" / "cli").is_dir():
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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
