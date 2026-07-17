"""Additional lifecycle coverage: install_signal_handlers and ValueError edge cases."""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from potpie_context_engine.adapters.outbound.daemon_process.pidfile import (
    install_signal_handlers,
    read_pid_file,
    remove_pid_file,
    write_pid_file,
)


def test_read_pid_file_returns_none_when_missing(tmp_path: pathlib.Path):
    assert read_pid_file(tmp_path / "nonexistent.pid") is None


def test_read_pid_file_returns_none_on_value_error(tmp_path: pathlib.Path):
    p = tmp_path / "bad.pid"
    p.write_text("not-a-number\n")
    assert read_pid_file(p) is None


def test_write_pid_file_overwrites_invalid_existing(tmp_path: pathlib.Path):
    p = tmp_path / "bad.pid"
    p.write_text("not-a-number\n")
    # existing = -1 branch — should not raise AlreadyRunning
    write_pid_file(p, 42)
    assert read_pid_file(p) == 42


def test_remove_pid_file_silent_on_missing(tmp_path: pathlib.Path):
    remove_pid_file(tmp_path / "ghost.pid")  # must not raise


@pytest.mark.anyio
async def test_install_signal_handlers_does_not_raise():
    """install_signal_handlers should succeed or silently ignore NotImplementedError/RuntimeError."""
    shutdown = asyncio.Event()
    install_signal_handlers(shutdown)
    # If we get here without exception, the function handled the platform gracefully.
    assert not shutdown.is_set()
