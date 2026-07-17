"""Deferred default pot during interactive setup."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie.services import setup_orchestrator
from potpie.services.host_wiring import build_host_shell
from potpie_context_core.domain.lifecycle import SKIPPED, SetupPlan


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_setup_run_skips_pot_default_when_deferred(host) -> None:
    report = host.setup.run(
        SetupPlan(repo="potpie", agent="claude", defer_default_pot=True),
    )
    steps = {s.step: s.state for s in report.steps}
    assert "pot.default" not in steps
    assert host.pots.active_pot() is None


def test_setup_run_defers_source_when_default_pot_deferred(host) -> None:
    existing = host.pots.create_pot(name="existing", use=True)

    report = host.setup.run(
        SetupPlan(repo="potpie", agent="claude", defer_default_pot=True),
    )

    steps = {s.step: s for s in report.steps}
    assert steps["source"].state == SKIPPED
    assert steps["source"].detail == "deferred until post-setup first pot"
    assert host.pots.list_sources(pot_id=existing.pot_id) == []


def test_setup_run_normalizes_repo_shorthand_before_source_registration(
    host, monkeypatch
) -> None:
    monkeypatch.setattr(
        setup_orchestrator,
        "_current_git_remote",
        lambda cwd: "github.com/acme/shop",
    )

    report = host.setup.run(SetupPlan(repo=".", pot="shop", agent="claude"))

    active = host.pots.active_pot()
    assert active is not None
    sources = host.pots.list_sources(pot_id=active.pot_id)
    assert sources[0].location == "github.com/acme/shop"
    source_step = next(step for step in report.steps if step.step == "source")
    assert source_step.detail == "registered repo 'github.com/acme/shop'"


def test_setup_preview_omits_pot_default_when_deferred(host) -> None:
    preview = host.setup.preview(SetupPlan(defer_default_pot=True))
    assert all(step.step != "pot.default" for step in preview.steps)


def test_setup_preview_marks_source_deferred_when_default_pot_deferred(host) -> None:
    preview = host.setup.preview(SetupPlan(defer_default_pot=True))
    source = next(step for step in preview.steps if step.step == "source")
    assert source.skip_reason == "deferred until post-setup first pot"


def test_setup_source_reason_matches_preview_when_repo_missing_and_pot_deferred(
    host,
) -> None:
    plan = SetupPlan(repo=None, defer_default_pot=True)

    preview = host.setup.preview(plan)
    report = host.setup.run(plan)

    preview_source = next(step for step in preview.steps if step.step == "source")
    run_source = next(step for step in report.steps if step.step == "source")
    assert preview_source.skip_reason == "no --repo provided"
    assert run_source.detail == "no --repo provided"
