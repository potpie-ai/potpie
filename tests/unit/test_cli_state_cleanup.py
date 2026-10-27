"""CLI test cleanup must remain effective when resource shutdown fails."""

# ruff: noqa: S101 - pytest tests use assertions intentionally.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from potpie_context_engine import Failure, Success

from potpie.cli.commands import _common
from potpie.config import host_registry
from potpie.runtime.resource_manager import ResourceLifecycleError
from tests.support.cli_state import reset_cli_state

pytestmark = pytest.mark.unit


def _populate_state(failures: set[str]) -> tuple[list[str], dict[str, object]]:
    calls: list[str] = []

    def action(name: str) -> None:
        calls.append(name)
        if name in failures:
            raise RuntimeError(f"{name} failed")

    shutdown = object()

    def run(awaitable: object) -> Success[None]:
        assert awaitable is shutdown
        action("shutdown")
        return Success(None)

    runtime = SimpleNamespace(close=lambda: action("runtime"))
    _common._state.update({key: object() for key in _common._new_cli_state()})
    old_services = {"cached": object()}
    _common._state.update(
        engine_runner=SimpleNamespace(run=run, close=lambda: action("runner")),
        engine_manager=SimpleNamespace(shutdown=lambda: shutdown),
        runtime=runtime,
        engine_runtime=runtime,
        root_runtime=runtime,
        root_product_services=old_services,
        engine_coordinator=object(),
        engine_remote_host=object(),
        injected_by_test=object(),
    )
    host_registry._current["origin"] = "managed"
    host_registry._warned_insecure.add("http://example.invalid")
    return calls, old_services


@pytest.mark.parametrize(
    "failures",
    [set(), {"shutdown"}, {"runner"}, {"runtime"}, {"shutdown", "runner", "runtime"}],
    ids=["success", "shutdown", "runner", "runtime", "all-fail"],
)
def test_cleanup_attempts_all_resources_and_restores_state(failures: set[str]) -> None:
    state = _common._state
    calls, old_services = _populate_state(failures)

    if failures:
        with pytest.raises(ExceptionGroup, match="CLI test cleanup failed") as caught:
            reset_cli_state()
        assert [str(exc) for exc in caught.value.exceptions] == [
            f"{name} failed"
            for name in ("shutdown", "runner", "runtime")
            if name in failures
        ]
        assert all(exc.__notes__ for exc in caught.value.exceptions)
    else:
        reset_cli_state()

    assert calls == ["shutdown", "runner", "runtime"]
    assert _common._state is state
    assert state == _common._new_cli_state()
    assert state["root_product_services"] is not old_services
    assert host_registry._current["origin"] is None
    assert host_registry._warned_insecure == set()

    reset_cli_state()
    assert calls == ["shutdown", "runner", "runtime"]


def test_state_defaults_have_independent_mutable_containers() -> None:
    first = _common._new_cli_state()
    second = _common._new_cli_state()
    first["root_product_services"]["service"] = object()
    assert second["root_product_services"] == {}


def test_cleanup_reports_manager_shutdown_failure_outcome() -> None:
    calls, _ = _populate_state(set())
    original_run = _common._state["engine_runner"].run
    failure = Failure(
        ResourceLifecycleError(
            code="resource_manager_shutdown_failed",
            message="cached resource failed to close",
            details={"failures": ({"resource": "index", "error": "close failed"},)},
        )
    )

    def run(awaitable: object) -> Failure[ResourceLifecycleError]:
        original_run(awaitable)
        return failure

    _common._state["engine_runner"].run = run
    with pytest.raises(ExceptionGroup, match="CLI test cleanup failed") as caught:
        reset_cli_state()

    assert calls == ["shutdown", "runner", "runtime"]
    error = caught.value.exceptions[0]
    assert "resource_manager_shutdown_failed" in str(error)
    assert "cached resource failed to close" in str(error)
    assert "index" in str(error)
    assert "close failed" in str(error)
    assert _common._state == _common._new_cli_state()
    assert host_registry._current["origin"] is None
    assert host_registry._warned_insecure == set()


def test_host_registry_failure_is_reported_after_cli_state_is_restored(monkeypatch):
    def fail_reset() -> None:
        raise RuntimeError("host registry failed")

    _common._state["injected_by_test"] = object()
    with monkeypatch.context() as patch:
        patch.setattr(host_registry, "reset_for_tests", fail_reset)
        with pytest.raises(ExceptionGroup, match="CLI test cleanup failed") as caught:
            reset_cli_state()

    assert str(caught.value.exceptions[0]) == "host registry failed"
    assert _common._state == _common._new_cli_state()


def test_fixture_reports_teardown_error_without_hiding_test_failure(tmp_path: Path):
    # Import the real autouse fixture into a standalone suite. Its two test
    # phases must remain separately visible, and the next test must be clean.
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import _reset_cli_state\n", encoding="utf-8"
    )
    (tmp_path / "test_cleanup.py").write_text(
        """from types import SimpleNamespace
from potpie.cli.commands import _common

def test_original_failure():
    def fail_close():
        raise RuntimeError("resource close failed")
    _common._state["runtime"] = SimpleNamespace(close=fail_close)
    _common._state["leaked"] = object()
    assert False, "original test failure"

def test_next_test_is_clean():
    assert _common._state == _common._new_cli_state()
""",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo), env.get("PYTHONPATH")])
    )
    result = subprocess.run(  # noqa: S603 - run pytest on the generated local suite.
        [sys.executable, "-m", "pytest", "-q", "--tb=short", str(tmp_path)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "original test failure" in output
    assert "resource close failed" in output
    assert "1 failed, 1 passed, 1 error" in output
