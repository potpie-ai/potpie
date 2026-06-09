"""Deferred global skills during interactive setup."""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from bootstrap.host_wiring import build_host_shell
from domain.lifecycle import SetupPlan


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_setup_run_skips_skills_when_deferred(host) -> None:
    report = host.setup.run(
        SetupPlan(repo="potpie", agent="claude", defer_skills=True),
    )
    steps = {s.step: s.state for s in report.steps}
    assert "skills" not in steps
    assert host.skills.status(agent="claude").installed == ()


def test_setup_preview_omits_skills_when_deferred(host) -> None:
    preview = host.setup.preview(SetupPlan(defer_skills=True))
    assert all(step.step != "skills" for step in preview.steps)
