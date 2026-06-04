"""End-to-end tests for unified-exec sessions on the real Daytona backend.

Verifies the Codex-style session contract against a live sandbox: a fast
command finishes within the yield window; a slow command yields while running
and progress is read incrementally; stdin is fed to an interactive command;
kill terminates a long runner; and a PTY session runs a command in a real
terminal. Uses the same local Daytona dev stack and ``daytona_service``
fixture as ``test_daytona_e2e.py``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.errors import ExecSessionNotFound
from sandbox.domain.models import (
    RepoIdentity,
    SessionExecRequest,
    SessionInputRequest,
    WorkspaceMode,
    WorkspaceRequest,
)

PUBLIC_REPO_URL = os.getenv(
    "DAYTONA_TEST_REPO_URL", "https://github.com/octocat/Hello-World.git"
)
PUBLIC_REPO_NAME = os.getenv("DAYTONA_TEST_REPO_NAME", "octocat/Hello-World")
PUBLIC_REPO_REF = os.getenv("DAYTONA_TEST_REPO_REF", "master")


def _request(conversation_id: str) -> WorkspaceRequest:
    # Distinct user/project so the derived Daytona sandbox name doesn't
    # truncate-collide with test_daytona_e2e.py's "potpie-e2e" sandbox. All
    # five tests share this (user, project) so the sandbox is created once and
    # reused via recovery; each gets its own worktree via conversation_id.
    return WorkspaceRequest(
        user_id="ptsess",
        project_id="ptsess",
        repo=RepoIdentity(repo_name=PUBLIC_REPO_NAME, repo_url=PUBLIC_REPO_URL),
        base_ref=PUBLIC_REPO_REF,
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
        create_branch=True,
    )


async def _safe_destroy(service: SandboxService, workspace_id: str) -> None:
    try:
        await service.destroy_workspace(workspace_id)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: destroy_workspace failed: {exc!r}")


@pytest.mark.asyncio
async def test_daytona_session_fast_command(daytona_service: SandboxService) -> None:
    ws = await daytona_service.get_or_create_workspace(_request("sess-fast"))
    try:
        result = await daytona_service.exec_session_start(
            ws.id,
            SessionExecRequest(cmd=("echo session-hello",), shell=True, yield_time_ms=5000),
        )
        assert result.running is False
        assert result.exit_code == 0
        assert "session-hello" in result.output
    finally:
        await _safe_destroy(daytona_service, ws.id)


@pytest.mark.asyncio
async def test_daytona_session_slow_command_progress(
    daytona_service: SandboxService,
) -> None:
    ws = await daytona_service.get_or_create_workspace(_request("sess-slow"))
    try:
        started = await daytona_service.exec_session_start(
            ws.id,
            SessionExecRequest(
                cmd=("echo first; sleep 2; echo second",),
                shell=True,
                yield_time_ms=400,
            ),
        )
        assert started.running is True
        assert "first" in started.output

        seen = started.output
        for _ in range(40):
            polled = await daytona_service.exec_session_poll(
                ws.id, started.session_id, yield_time_ms=400
            )
            seen += polled.output
            if not polled.running:
                assert polled.exit_code == 0
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("session never finished")
        assert "second" in seen
    finally:
        await _safe_destroy(daytona_service, ws.id)


@pytest.mark.asyncio
async def test_daytona_session_write_stdin(daytona_service: SandboxService) -> None:
    ws = await daytona_service.get_or_create_workspace(_request("sess-stdin"))
    session_id = None
    try:
        started = await daytona_service.exec_session_start(
            ws.id,
            SessionExecRequest(cmd=("cat",), shell=True, yield_time_ms=400),
        )
        session_id = started.session_id
        assert started.running is True
        wrote = await daytona_service.exec_session_write(
            ws.id,
            SessionInputRequest(
                session_id=started.session_id, data="ping-pong\n", yield_time_ms=1000
            ),
        )
        assert "ping-pong" in wrote.output
    finally:
        if session_id is not None:
            await daytona_service.exec_session_kill(ws.id, session_id)
        await _safe_destroy(daytona_service, ws.id)


@pytest.mark.asyncio
async def test_daytona_session_kill(daytona_service: SandboxService) -> None:
    ws = await daytona_service.get_or_create_workspace(_request("sess-kill"))
    try:
        started = await daytona_service.exec_session_start(
            ws.id,
            SessionExecRequest(cmd=("sleep 60",), shell=True, yield_time_ms=400),
        )
        assert started.running is True
        await daytona_service.exec_session_kill(ws.id, started.session_id)
        with pytest.raises(ExecSessionNotFound):
            await daytona_service.exec_session_poll(
                ws.id, started.session_id, yield_time_ms=200
            )
    finally:
        await _safe_destroy(daytona_service, ws.id)


@pytest.mark.asyncio
async def test_daytona_session_pty(daytona_service: SandboxService) -> None:
    ws = await daytona_service.get_or_create_workspace(_request("sess-pty"))
    session_id = None
    try:
        started = await daytona_service.exec_session_start(
            ws.id,
            SessionExecRequest(
                cmd=("echo pty-marker",), shell=True, tty=True, yield_time_ms=2000
            ),
        )
        session_id = started.session_id
        seen = started.output
        for _ in range(20):
            if "pty-marker" in seen:
                break
            polled = await daytona_service.exec_session_poll(
                ws.id, session_id, yield_time_ms=500
            )
            seen += polled.output
        assert "pty-marker" in seen
    finally:
        if session_id is not None:
            await daytona_service.exec_session_kill(ws.id, session_id)
        await _safe_destroy(daytona_service, ws.id)
