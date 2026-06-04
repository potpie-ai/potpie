"""Unified-exec session behavior on the local subprocess runtime.

Exercises the Codex-style contract end to end at the provider level: a fast
command finishes within the yield window; a slow command yields while still
running and its progress is read incrementally; stdin is fed to an
interactive reader; and kill terminates a long-running command.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.domain.errors import ExecSessionNotFound
from sandbox.domain.models import (
    Mount,
    RuntimeSpec,
    SessionExecRequest,
    SessionInputRequest,
)


def _spec(root: Path) -> RuntimeSpec:
    return RuntimeSpec(
        image=None,
        workdir=str(root),
        mounts=(Mount(source=str(root), target=str(root), writable=True),),
    )


async def _runtime(tmp_path: Path):
    provider = LocalSubprocessRuntimeProvider()
    runtime = await provider.create("ws_test", _spec(tmp_path))
    return provider, runtime


@pytest.mark.asyncio
async def test_fast_command_finishes_within_yield_window(tmp_path: Path) -> None:
    provider, runtime = await _runtime(tmp_path)
    result = await provider.exec_session_start(
        runtime,
        SessionExecRequest(cmd=("echo hello",), shell=True, yield_time_ms=2000),
    )
    assert result.running is False
    assert result.exit_code == 0
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_slow_command_yields_then_reports_progress(tmp_path: Path) -> None:
    provider, runtime = await _runtime(tmp_path)
    # Emit a line, sleep, emit another. With a 300ms window the first start
    # returns still-running with only the first line; a poll picks up the rest.
    script = "echo first; sleep 0.6; echo second"
    started = await provider.exec_session_start(
        runtime,
        SessionExecRequest(cmd=(script,), shell=True, yield_time_ms=300),
    )
    assert started.running is True
    assert "first" in started.output
    assert "second" not in started.output

    # Poll until the command finishes; accumulate output.
    seen = started.output
    for _ in range(20):
        polled = await provider.exec_session_poll(
            runtime, started.session_id, yield_time_ms=300
        )
        seen += polled.output
        if not polled.running:
            assert polled.exit_code == 0
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("command never finished")
    assert "second" in seen


@pytest.mark.asyncio
async def test_write_stdin_feeds_interactive_command(tmp_path: Path) -> None:
    provider, runtime = await _runtime(tmp_path)
    # `cat` echoes whatever it receives on stdin until EOF.
    started = await provider.exec_session_start(
        runtime,
        SessionExecRequest(cmd=("cat",), shell=True, yield_time_ms=200),
    )
    assert started.running is True

    wrote = await provider.exec_session_write(
        runtime,
        SessionInputRequest(
            session_id=started.session_id, data="ping\n", yield_time_ms=400
        ),
    )
    assert "ping" in wrote.output
    assert wrote.running is True

    # Closing stdin (EOF) lets cat exit.
    await provider.exec_session_kill(runtime, started.session_id)
    with pytest.raises(ExecSessionNotFound):
        await provider.exec_session_poll(
            runtime, started.session_id, yield_time_ms=50
        )


@pytest.mark.asyncio
async def test_kill_terminates_long_running_command(tmp_path: Path) -> None:
    provider, runtime = await _runtime(tmp_path)
    started = await provider.exec_session_start(
        runtime,
        SessionExecRequest(cmd=("sleep 30",), shell=True, yield_time_ms=200),
    )
    assert started.running is True
    await provider.exec_session_kill(runtime, started.session_id)
    with pytest.raises(ExecSessionNotFound):
        await provider.exec_session_poll(
            runtime, started.session_id, yield_time_ms=50
        )


@pytest.mark.asyncio
async def test_pty_mode_runs_command(tmp_path: Path) -> None:
    provider, runtime = await _runtime(tmp_path)
    result = await provider.exec_session_start(
        runtime,
        SessionExecRequest(
            cmd=("echo tty-here",), shell=True, tty=True, yield_time_ms=2000
        ),
    )
    assert result.running is False
    assert result.exit_code == 0
    assert "tty-here" in result.output
