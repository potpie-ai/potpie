from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from potpie.cli.commands import _common
from potpie.runtime.async_bridge import run_sync
from potpie.runtime.composition import PotpieRuntime


@pytest.fixture(autouse=True)
def _shutdown_runner() -> Iterator[None]:
    yield
    _common.set_cli_runtime(None)
    from potpie.runtime import reset_runtime

    reset_runtime()


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


def test_run_sync_reuses_loop_and_constructs_operation_on_runner_thread() -> None:
    factory_threads: list[int] = []

    def factory():
        factory_threads.append(threading.get_ident())

        async def operation() -> tuple[int, int]:
            return id(asyncio.get_running_loop()), threading.get_ident()

        return operation()

    first = run_sync(factory)
    second = run_sync(factory)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert factory_threads == [first[1], first[1]]


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
    injected = cast(PotpieRuntime, object())

    def unexpected_root_runtime() -> Any:
        raise AssertionError("root runtime should not be loaded")

    monkeypatch.setattr("potpie.runtime.get_runtime", unexpected_root_runtime)
    _common.set_cli_runtime(injected)

    assert _common.get_cli_runtime() is injected


def test_cli_runtime_uses_root_singleton_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_runtime = cast(PotpieRuntime, object())
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
    _common.set_cli_runtime(cast(PotpieRuntime, object()))

    _common.reset_cli_runtime()

    assert reset_calls == 1
    replacement = cast(PotpieRuntime, object())
    monkeypatch.setattr("potpie.runtime.get_runtime", lambda: replacement)
    assert _common.get_cli_runtime() is replacement


def test_runtime_is_constructed_and_closed_on_the_runner_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from potpie.runtime import composition

    events: list[tuple[str, int]] = []

    class Runtime:
        async def aclose(self) -> None:
            events.append(("close", id(asyncio.get_running_loop())))

    def build_runtime(*, runtime_override: str | None = None) -> Any:
        del runtime_override
        events.append(("create", id(asyncio.get_running_loop())))
        return Runtime()

    monkeypatch.setattr(composition, "create_runtime", build_runtime)

    runtime = composition.get_runtime()
    operation_loop = run_sync(_current_loop_id)
    composition.reset_runtime()

    assert runtime is not None
    assert events == [("create", operation_loop), ("close", operation_loop)]


def test_concurrent_runtime_callers_create_one_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from potpie.runtime import composition

    created = 0

    class Runtime:
        async def aclose(self) -> None:
            return None

    def build_runtime(*, runtime_override: str | None = None) -> Any:
        nonlocal created
        del runtime_override
        created += 1
        return Runtime()

    monkeypatch.setattr(composition, "create_runtime", build_runtime)
    with ThreadPoolExecutor(max_workers=8) as executor:
        runtimes = list(
            executor.map(lambda _index: composition.get_runtime(), range(24))
        )

    assert created == 1
    assert all(runtime is runtimes[0] for runtime in runtimes)


async def _current_loop_id() -> int:
    return id(asyncio.get_running_loop())


@pytest.mark.parametrize("entrypoint_module", ("potpie.cli.main", "potpie.mcp.main"))
def test_process_entrypoints_always_reset_runtime(
    entrypoint_module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from potpie import runtime as runtime_module

    module = importlib.import_module(entrypoint_module)
    reset_calls = 0

    def reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(runtime_module, "reset_runtime", reset)
    if entrypoint_module == "potpie.cli.main":
        monkeypatch.setattr(module, "run_cli", lambda: None)
    else:
        monkeypatch.setattr(module.mcp, "run", lambda: None)

    module.main()

    assert reset_calls == 1
