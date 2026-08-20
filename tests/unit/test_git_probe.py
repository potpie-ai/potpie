"""One git probe for the three sites that used to each have their own."""

from __future__ import annotations

import subprocess
from pathlib import Path

from potpie_context_engine.domain.git_probe import current_git_remote, run_git_probe


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_probe_answers_inside_a_repository_and_not_outside(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    assert run_git_probe(["rev-parse", "--is-inside-work-tree"], cwd=repo) == "true"
    # Not a repository: git exits non-zero, the probe answers None, nothing raises.
    assert run_git_probe(["rev-parse", "--is-inside-work-tree"], cwd=tmp_path / "plain") is None
    assert run_git_probe(["rev-parse", "HEAD"], cwd=tmp_path / "does-not-exist") is None


def test_current_git_remote_is_normalized_to_the_pot_store_key(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    assert current_git_remote(repo) is None
    _git(repo, "remote", "add", "origin", "git@github.com:Acme/Shop.git")
    assert current_git_remote(repo) == "github.com/acme/shop"


def test_the_three_callers_share_it(monkeypatch) -> None:
    """The fix lived in one site and the other two kept the shape that wedges
    on Windows; now there is one shape. Patching the shared probe reaches all."""
    import potpie_context_engine.domain.git_probe as probe
    from potpie.cli import repo_location
    from potpie.cli.commands import graph
    from potpie_context_engine.application.services import setup_orchestrator

    monkeypatch.setattr(probe, "run_git_probe", lambda *a, **k: "https://github.com/Acme/Shop.git")
    assert repo_location.current_git_remote(Path(".")) == "github.com/acme/shop"
    assert setup_orchestrator._current_git_remote(Path(".")) == "github.com/acme/shop"
    assert graph._current_repo_remote_for_scope() == "github.com/acme/shop"
