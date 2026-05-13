"""Sandbox metrics sink — swappable, safe to call, default no-op."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

pytestmark = pytest.mark.unit


def test_default_sink_does_not_raise() -> None:
    from adapters.outbound.agent_tools._sandbox_metrics import record

    # Default sink logs at debug; no exception even with weird types.
    record("pot_sandbox.tool_call", {"name": "x", "ok": False}, value=2.5)
    record("pot_sandbox.attach")  # labels default {}


def test_set_sink_swaps_then_restores() -> None:
    from adapters.outbound.agent_tools import _sandbox_metrics as mod

    captured: list[tuple[str, Mapping[str, Any], float]] = []

    def _sink(name: str, labels: Mapping[str, Any], value: float) -> None:
        captured.append((name, dict(labels), value))

    mod.set_sink(_sink)
    try:
        mod.record("pot_sandbox.checkout", {"repo": "a/b"}, value=1)
        mod.record("pot_sandbox.cold_start_ms", {"repo": "a/b"}, value=850)
    finally:
        mod.set_sink(None)
    assert captured == [
        ("pot_sandbox.checkout", {"repo": "a/b"}, 1.0),
        ("pot_sandbox.cold_start_ms", {"repo": "a/b"}, 850.0),
    ]
    # After restore, the default sink is back and still safe.
    mod.record("pot_sandbox.tool_call", {"name": "x", "ok": True})


def test_sink_exceptions_are_swallowed() -> None:
    from adapters.outbound.agent_tools import _sandbox_metrics as mod

    def _broken(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("telemetry backend down")

    mod.set_sink(_broken)
    try:
        # Must not raise.
        mod.record("pot_sandbox.attach", {"x": 1})
    finally:
        mod.set_sink(None)
