from __future__ import annotations

import pickle
import subprocess

# ruff: noqa: S101, S301 - test assertions and private worker pickle protocol.

import pytest

from potpie.runtime import (
    AuthenticatedDaemonCaller,
    ContextSelector,
    EngineOperation,
    EngineOperationRequest,
    PROTOCOL_VERSION,
)
from potpie_context_engine import Success
from potpie_context_engine.requests import SearchRequest


class _FallbackHandler:
    async def handle(self, request, *, authentication):
        del request, authentication
        return Success("fallback")


def _request() -> EngineOperationRequest:
    return EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="windows-worker-test",
        operation=EngineOperation.SEARCH,
        selector=ContextSelector(kind="explicit", value="pot-test"),
        payload=SearchRequest(query="query"),
    )


@pytest.mark.anyio
async def test_windows_ladybug_handler_isolates_engine_operation(monkeypatch) -> None:
    from potpie.daemon import windows_worker

    expected = Success("worker-result")
    captured = {}

    async def fake_to_thread(function, *args):
        if function is windows_worker._close_backend_connections:
            captured["close_function"] = function
            return None
        captured["function"] = function
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=pickle.dumps(expected), stderr=b""
        )

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.asyncio, "to_thread", fake_to_thread)

    handler = windows_worker.WindowsLadybugOperationHandler(
        fallback=_FallbackHandler(), backend_profile="ladybug"
    )
    result = await handler.handle(
        _request(), authentication=AuthenticatedDaemonCaller()
    )

    assert result == expected
    assert captured["function"] is windows_worker._run_worker_process
    assert captured["args"][0] == _request()
    assert captured["args"][1]["CONTEXT_ENGINE_HOST_MODE"] == "in_process"


@pytest.mark.anyio
async def test_windows_worker_handler_isolates_search_for_other_profiles(
    monkeypatch,
) -> None:
    from potpie.daemon import windows_worker

    expected = Success("worker-result")

    async def fake_to_thread(function, *args):
        if function is windows_worker._close_backend_connections:
            return None
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=pickle.dumps(expected), stderr=b""
        )

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.asyncio, "to_thread", fake_to_thread)
    handler = windows_worker.WindowsLadybugOperationHandler(
        fallback=_FallbackHandler(), backend_profile="embedded"
    )

    result = await handler.handle(
        _request(), authentication=AuthenticatedDaemonCaller()
    )

    assert result == expected


@pytest.mark.anyio
async def test_windows_worker_handler_normalizes_ladybug_profile(monkeypatch) -> None:
    from potpie.daemon import windows_worker

    expected = Success("worker-result")

    async def fake_to_thread(function, *args):
        if function is windows_worker._close_backend_connections:
            return None
        assert function is windows_worker._run_worker_process
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=pickle.dumps(expected), stderr=b""
        )

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.asyncio, "to_thread", fake_to_thread)
    handler = windows_worker.WindowsLadybugOperationHandler(
        fallback=_FallbackHandler(), backend_profile=" Ladybug "
    )

    result = await handler.handle(
        _request(), authentication=AuthenticatedDaemonCaller()
    )

    assert result == expected


@pytest.mark.anyio
async def test_windows_worker_closes_parent_ladybug_provider(monkeypatch) -> None:
    from potpie.daemon import windows_worker

    expected = Success("worker-result")
    calls = []

    class _Provider:
        def close(self):
            calls.append("close")

    class _Backend:
        graph_provider = _Provider()

    async def fake_to_thread(function, *args):
        calls.append(function)
        if function is windows_worker._close_backend_connections:
            function(*args)
            return None
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=pickle.dumps(expected), stderr=b""
        )

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.asyncio, "to_thread", fake_to_thread)
    handler = windows_worker.WindowsLadybugOperationHandler(
        fallback=_FallbackHandler(), backend_profile="ladybug", backend=_Backend()
    )

    result = await handler.handle(
        _request(), authentication=AuthenticatedDaemonCaller()
    )

    assert result == expected
    assert calls[0] is windows_worker._close_backend_connections
    assert "close" in calls


@pytest.mark.anyio
async def test_windows_worker_native_exit_becomes_typed_failure(monkeypatch) -> None:
    from potpie.daemon import windows_worker

    async def fake_to_thread(function, *args):
        if function is windows_worker._close_backend_connections:
            return None
        assert function is windows_worker._run_worker_process
        del args
        return subprocess.CompletedProcess(
            args=(), returncode=-1073741819, stdout=b"", stderr=b"native crash"
        )

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.asyncio, "to_thread", fake_to_thread)
    handler = windows_worker.WindowsLadybugOperationHandler(
        fallback=_FallbackHandler(), backend_profile="ladybug"
    )

    result = await handler.handle(
        _request(), authentication=AuthenticatedDaemonCaller()
    )

    assert result.error.code == "daemon_worker_failed"
    assert result.error.details["reason"] == "exit_-1073741819"


@pytest.mark.unit
def test_windows_worker_process_uses_standard_subprocess(monkeypatch) -> None:
    from potpie.daemon import windows_worker

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, b"out", b"err")

    monkeypatch.setattr(windows_worker.sys, "platform", "win32")
    monkeypatch.setattr(windows_worker.subprocess, "run", fake_run)
    result = windows_worker._run_worker_process(
        _request(), {"CONTEXT_ENGINE_HOME": "C:\\qa"}, 12.0
    )

    assert result.returncode == 0
    assert captured["command"] == [
        windows_worker.sys.executable,
        "-m",
        "potpie.daemon.windows_worker",
        "--worker",
    ]
    assert captured["kwargs"]["timeout"] == 12.0
    assert captured["kwargs"]["creationflags"] == 0x08000000
    assert pickle.loads(captured["kwargs"]["input"]) == _request()
