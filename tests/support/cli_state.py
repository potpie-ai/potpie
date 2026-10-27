"""Resource cleanup and state isolation for in-process CLI tests."""

from collections.abc import Callable

from potpie_context_engine import Failure

from potpie.cli.commands import _common
from potpie.config import host_registry


def reset_cli_state() -> None:
    """Attempt all owned cleanup, restore defaults, then report any failures."""
    runner = _common._state.get("engine_runner")
    manager = _common._state.get("engine_manager")
    runtime = _common._state.get("runtime")
    errors: list[Exception] = []

    def attempt(label: str, cleanup: Callable[[], object]) -> None:
        try:
            cleanup()
        except Exception as exc:
            exc.add_note(f"CLI test cleanup: {label}")
            errors.append(exc)

    def close_runtime() -> None:
        # The runtime owns its drain thread and index; cached services only
        # reference it and must not be closed separately.
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

    def shutdown_manager() -> None:
        outcome = runner.run(manager.shutdown())
        # Resource managers report failed engine/resource closes as outcomes;
        # those failures must be as visible as exceptions from shutdown itself.
        if isinstance(outcome, Failure):
            raise RuntimeError(f"engine manager shutdown failed: {outcome.error!r}")

    try:
        if runner is not None and manager is not None:
            attempt("engine manager shutdown", shutdown_manager)
        if runner is not None:
            attempt("engine runner close", runner.close)
        attempt("runtime close", close_runtime)
    finally:
        # Preserve references to the state dictionary, drop dynamic keys, and
        # allocate fresh mutable defaults for the next test.
        _common._state.clear()
        _common._state.update(_common._new_cli_state())
        attempt("host registry reset", host_registry.reset_for_tests)

    if errors:
        raise ExceptionGroup("CLI test cleanup failed", errors)
