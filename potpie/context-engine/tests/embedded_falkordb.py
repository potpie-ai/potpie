"""Private FalkorDB Lite servers for tests, always stopped afterwards.

``redislite`` daemonizes every ``redis-server`` it starts, so a server a test
forgets survives the whole run (and, on a CI runner, the job). Its ``close()``
stops the server only when the server reports a single client: a test that
queried from two threads, or through a pipeline, leaves a second pooled
connection behind, and ``close()`` then only disconnects. These helpers shut the
server down explicitly, whatever its pool holds.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def embedded_falkordb(path: Path | str) -> Iterator[Any]:
    """Start a FalkorDB Lite server on ``path`` and stop it on exit."""

    from redislite.falkordb_client import FalkorDB

    database = FalkorDB(str(path))
    try:
        yield database
    finally:
        stop_embedded_falkordb(database)


def stop_embedded_falkordb(database: Any) -> None:
    """Stop ``database``'s server even when several connections are open."""

    from redis.backoff import NoBackoff
    from redis.exceptions import RedisError
    from redis.retry import Retry

    connection = database.connection
    pid = connection.pid
    redis_dir = connection.redis_dir
    # The server answers SHUTDOWN by closing the connection. redis-py retries a
    # closed connection with backoff for seconds before accepting that; on a
    # local socket a retry cannot help.
    connection.set_retry(Retry(NoBackoff(), 0))
    connection.connection_pool.disconnect()
    try:
        connection.shutdown(now=True, force=True)
    except RedisError:
        pass
    _wait_for_exit(pid)
    # Releases redislite's bookkeeping; the server is already gone.
    database.close()
    if redis_dir:
        shutil.rmtree(redis_dir, ignore_errors=True)


def _wait_for_exit(pid: int | None, *, timeout: float = 5.0) -> None:
    if not pid:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
