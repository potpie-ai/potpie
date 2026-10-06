"""``potpie setup`` registers one repo source, however often it runs.

Setup promises an idempotent first run, and re-running it is the natural
response to a failure. The repo it registers is matched by identity key, so a
re-run (or the same repo spelled another way) never appends a second source,
and a re-run never takes back a repo default someone chose by hand.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import subprocess

import pytest

from potpie.pots.local_store import LocalPotStore
from potpie.runtime.composition import build_local_runtime
from potpie.setup import local_installer
from potpie.setup import orchestrator as setup_orchestrator
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.lifecycle import DONE, SKIPPED, SetupPlan, StepResult

pytestmark = pytest.mark.unit


@pytest.fixture()
def host(tmp_path, monkeypatch):
    """A real root runtime on a temp home: the pot store is what is under test."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    # The installer step probes the ``potpie`` on PATH; that is covered in
    # test_local_installer.py and is not what these tests are about.
    monkeypatch.setattr(
        local_installer, "probe_cli_surface", lambda executable: {"ok": True}
    )
    return build_local_runtime(backend=InMemoryGraphBackend()).root


def _plan(**overrides) -> SetupPlan:
    base = {
        "host_mode": "in_process",
        "backend": "in_memory",
        "repo": ".",
        "pot": "p",
        "agent": "default",
        "embeddings": "none",
    }
    return SetupPlan(**{**base, **overrides})


def _repo_sources(host, pot_id: str) -> list:
    return [s for s in host.pots.list_sources(pot_id=pot_id) if s.kind == "repo"]


def _step(report, name: str) -> StepResult:
    return next(s for s in report.steps if s.step == name)


def test_three_setup_runs_leave_one_repo_source(host, tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    # No remote: ``.`` resolves to the absolute working tree, the exact spelling
    # a comparison against the literal flag could never match.
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)

    reports = [host.setup.run(_plan()) for _ in range(3)]

    assert all(r.ok for r in reports)
    active = host.pots.active_pot()
    assert active is not None
    sources = _repo_sources(host, active.pot_id)
    assert len(sources) == 1, [s.location for s in sources]
    assert sources[0].location == str(repo.resolve())


def test_a_repeat_run_reports_the_source_step_as_skipped_with_the_existing_id(
    host, tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)

    first = host.setup.run(_plan())
    second = host.setup.run(_plan())

    registered = _step(first, "source")
    repeated = _step(second, "source")
    assert registered.state == DONE
    assert repeated.state == SKIPPED
    assert repeated.metadata["already_registered"] is True
    assert repeated.metadata["source_id"] == registered.metadata["source_id"]
    # A converged re-run is not a degraded run.
    assert second.to_dict()["ok"] is True


def test_the_same_repo_spelled_two_ways_is_one_source(host) -> None:
    first = host.setup.run(_plan(repo="git@github.com:Acme/Shop.git"))
    second = host.setup.run(_plan(repo="https://github.com/acme/shop/"))

    active = host.pots.active_pot()
    assert active is not None
    assert len(_repo_sources(host, active.pot_id)) == 1
    assert _step(first, "source").state == DONE
    assert _step(second, "source").state == SKIPPED


def test_setup_lower_cases_a_remote_the_way_source_add_does(
    host, tmp_path, monkeypatch
) -> None:
    """One repo, one spelling, through the real ``git remote get-url`` path."""
    repo = tmp_path / "mixedcase"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:Acme-Labs/Shop.git"],
        cwd=repo,
        check=True,
    )
    monkeypatch.chdir(repo)

    host.setup.run(_plan(repo="."))

    active = host.pots.active_pot()
    assert active is not None
    sources = _repo_sources(host, active.pot_id)
    assert [s.location for s in sources] == ["github.com/acme-labs/shop"]
    assert host.pots.repo_default(repo="github.com/acme-labs/shop") == active.pot_id


def test_a_repeat_run_does_not_steal_a_deliberate_repo_default(host) -> None:
    host.setup.run(_plan(repo="github.com/acme/shop"))
    other = host.pots.create_pot(name="chosen-by-hand")
    host.pots.set_repo_default(repo="github.com/acme/shop", pot_id=other.pot_id)

    host.setup.run(_plan(repo="github.com/acme/shop"))

    assert host.pots.repo_default(repo="github.com/acme/shop") == other.pot_id
    active = host.pots.active_pot()
    assert active is not None and active.pot_id != other.pot_id
    assert len(_repo_sources(host, active.pot_id)) == 1


def test_a_dangling_repo_default_is_still_repaired_by_a_repeat_run(host) -> None:
    # A run that stopped between add_source and the binding, or a binding left
    # pointing at a pot that no longer exists, must still converge.
    host.setup.run(_plan(repo="github.com/acme/shop"))
    # Straight to the store: the service refuses to bind a pot that is gone,
    # which is precisely the state left behind when one is removed afterwards.
    LocalPotStore().set_repo_default(repo="github.com/acme/shop", pot_id="pot_deleted")

    host.setup.run(_plan(repo="github.com/acme/shop"))

    active = host.pots.active_pot()
    assert active is not None
    assert host.pots.repo_default(repo="github.com/acme/shop") == active.pot_id
