from __future__ import annotations

from typing import Any

import pytest

from potpie.cli.commands import _common
from potpie.runtime.async_bridge import run_sync


def test_run_sync_executes_operation_factory() -> None:
    constructed = 0

    async def operation() -> int:
        return 42

    def factory():
        nonlocal constructed
        constructed += 1
        return operation()

    assert run_sync(factory) == 42
    assert constructed == 1


def test_run_sync_propagates_operation_exception() -> None:
    expected = LookupError("operation failed")

    async def operation() -> None:
        raise expected

    with pytest.raises(LookupError) as caught:
        run_sync(operation)

    assert caught.value is expected


@pytest.mark.asyncio
async def test_run_sync_rejects_running_loop_before_constructing_coroutine() -> None:
    constructed = False

    async def operation() -> None:
        return None

    def factory():
        nonlocal constructed
        constructed = True
        return operation()

    with pytest.raises(RuntimeError, match="inside an event loop"):
        run_sync(factory)

    assert constructed is False


def test_cli_runtime_prefers_explicit_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = object()

    def unexpected_root_runtime() -> Any:
        raise AssertionError("root runtime should not be loaded")

    monkeypatch.setattr("potpie.runtime.get_runtime", unexpected_root_runtime)
    _common.set_cli_runtime(injected)

    assert _common.get_cli_runtime() is injected


def test_cli_runtime_uses_root_singleton_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_runtime = object()
    monkeypatch.setattr("potpie.runtime.get_runtime", lambda: root_runtime)
    _common.set_cli_runtime(None)

    assert _common.get_cli_runtime() is root_runtime


def test_reset_cli_runtime_clears_injection_and_root_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_calls = 0

    def reset_root_runtime() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr("potpie.runtime.reset_runtime", reset_root_runtime)
    _common.set_cli_runtime(object())

    _common.reset_cli_runtime()

    assert reset_calls == 1
    replacement = object()
    monkeypatch.setattr("potpie.runtime.get_runtime", lambda: replacement)
    assert _common.get_cli_runtime() is replacement
