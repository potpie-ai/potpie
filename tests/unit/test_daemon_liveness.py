"""``pid_alive`` asks the kernel, not ``os.kill(pid, 0)``."""

from __future__ import annotations

import subprocess
import sys

from potpie.daemon.process.liveness import pid_alive


def test_a_live_process_is_alive_and_a_reaped_one_is_not() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert pid_alive(proc.pid) is False


def test_a_zombie_is_not_alive() -> None:
    """Killed but not yet reaped: it still owns its pid, and ``os.kill(pid, 0)``
    would say yes. Nothing can talk to it, so it is not a running daemon."""
    import time

    import psutil

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # Let it exit without reaping it (no wait()), then ask the kernel.
        for _ in range(100):
            if psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE
        assert pid_alive(proc.pid) is False
    finally:
        proc.wait(timeout=10)


def test_non_positive_pids_are_never_alive() -> None:
    assert pid_alive(0) is False
    assert pid_alive(-1) is False
