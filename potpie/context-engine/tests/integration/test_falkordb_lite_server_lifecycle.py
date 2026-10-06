"""Lifecycle of the embedded FalkorDB Lite server, against a real server.

The embedded server is a separate, daemonized ``redis-server``. These pin how a
process lets go of it: quickly, so ``potpie daemon stop`` does not time out,
and never out from under another process that is still using it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

pytest.importorskip("redislite.falkordb_client")
psutil = pytest.importorskip("psutil")

from potpie_context_engine.adapters.outbound.graph import (  # noqa: E402
    falkordb_writer,
)

pytestmark = pytest.mark.integration


class _LiteSettings:
    def __init__(self, path: str) -> None:
        self._path = path

    def falkordb_graph_name(self) -> str:
        return "context_graph"

    def falkordb_mode(self) -> str:
        return "lite"

    def falkordb_lite_path(self) -> str:
        return self._path


def _server_pid(path: str) -> int:
    with open(path + ".settings") as handle:
        pidfile = json.load(handle)["pidfile"]
    with open(pidfile) as handle:
        return int(handle.read().strip())


def _wait_until_gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.05)
    return not psutil.pid_exists(pid)


@pytest.fixture()
def lite_db(tmp_path):
    """A db path, plus the server pids a test started, stopped afterwards."""
    falkordb_writer._OPEN_SERVERS.clear()
    started: list[int] = []
    yield str(tmp_path / "context_graph" / "falkordb.db"), started
    falkordb_writer._OPEN_SERVERS.clear()
    for pid in started:
        try:
            psutil.Process(pid).terminate()
        except psutil.NoSuchProcess:
            continue
        _wait_until_gone(pid)


def test_stopping_the_server_does_not_wait_out_connection_retries(lite_db) -> None:
    # The server answers SHUTDOWN by dropping the connection. With redis-py's
    # default retry policy that expected error is retried with backoff for
    # several seconds, which made `potpie daemon stop` time out.
    path, started = lite_db
    graph = falkordb_writer.build_falkordb_graph(_LiteSettings(path))
    graph.query("RETURN 1")
    pid = _server_pid(path)
    started.append(pid)

    began = time.monotonic()
    assert falkordb_writer.shutdown_embedded_servers() == 1
    elapsed = time.monotonic() - began

    assert elapsed < 2.0, f"stopping the embedded server took {elapsed:.1f}s"
    assert _wait_until_gone(pid)


# The process that starts the server: it opens the graph, reports the server,
# waits until told to go, then exits normally so its atexit hooks run.
_STARTER = """
import json, sys
from potpie_context_engine.adapters.outbound.graph import falkordb_writer

class Settings:
    def falkordb_graph_name(self): return "context_graph"
    def falkordb_mode(self): return "lite"
    def falkordb_lite_path(self): return sys.argv[1]

falkordb_writer.build_falkordb_graph(Settings()).query("RETURN 1")
print("ready", flush=True)
sys.stdin.readline()
"""


def test_a_server_is_handed_over_to_the_process_still_using_it(lite_db) -> None:
    # `potpie setup` starts the server while provisioning, launches the daemon,
    # and exits while the daemon is attached. The daemon must keep its server,
    # and stop it when it is the last one to let go.
    path, started = lite_db
    starter = subprocess.Popen(
        [sys.executable, "-c", _STARTER, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert starter.stdout is not None and starter.stdin is not None
        assert starter.stdout.readline().strip() == "ready"
        pid = _server_pid(path)
        started.append(pid)

        attached = falkordb_writer.build_falkordb_graph(_LiteSettings(path))
        attached.query("RETURN 1")
        starter.stdin.write("exit\n")
        starter.stdin.flush()
        assert starter.wait(timeout=30) == 0
    finally:
        if starter.poll() is None:
            starter.kill()

    assert psutil.pid_exists(pid), "the starting process stopped a server in use"
    attached.query("RETURN 1")

    assert falkordb_writer.shutdown_embedded_servers() == 1
    assert _wait_until_gone(pid)
