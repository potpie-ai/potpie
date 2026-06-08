"""Deferred default pot during interactive setup."""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from bootstrap.host_wiring import build_host_shell
from domain.lifecycle import SetupPlan


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_setup_run_skips_pot_default_when_deferred(host) -> None:
    report = host.setup.run(
        SetupPlan(repo="potpie", agent="claude", defer_default_pot=True),
    )
    steps = {s.step: s.state for s in report.steps}
    assert "pot.default" not in steps
    assert host.pots.active_pot() is None


def test_setup_preview_omits_pot_default_when_deferred(host) -> None:
    preview = host.setup.preview(SetupPlan(defer_default_pot=True))
    assert all(step.step != "pot.default" for step in preview.steps)
