"""Tests for pot-wide vs repo-targeted backfill."""

from __future__ import annotations

from unittest.mock import MagicMock

from application.use_cases.backfill_pot import backfill_pot_context
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo


def _repo(pot_id: str, repo_name: str, *, rid: str | None = None) -> ResolvedPotRepo:
    return ResolvedPotRepo(
        pot_id=pot_id,
        repo_id=rid or f"rid-{repo_name}",
        provider="github",
        provider_host="github.com",
        repo_name=repo_name,
        ready=True,
    )


def test_backfill_pot_wide_runs_all_repos() -> None:
    settings = MagicMock()
    settings.is_enabled.return_value = True
    settings.backfill_max_prs_per_run.return_value = 10

    r1 = _repo("p1", "o/r1")
    r2 = _repo("p1", "o/r2")
    resolved = ResolvedPot(pot_id="p1", name="p1", repos=[r1, r2], ready=True)

    pots = MagicMock()
    pots.resolve_pot.return_value = resolved

    repos_called: list[str] = []

    def source_for_repo(name: str) -> MagicMock:
        repos_called.append(name)
        src = MagicMock()
        src.iter_closed_pulls.return_value = []
        return src

    ledger = MagicMock()
    sync = MagicMock()
    sync.last_synced_at = None
    ledger.get_or_create_sync_state.return_value = sync

    context_graph = MagicMock()

    out = backfill_pot_context(
        settings,
        pots,
        source_for_repo,
        ledger,
        context_graph,
        "p1",
        target_repo_name=None,
    )

    assert set(repos_called) == {"o/r1", "o/r2"}
    assert out["status"] == "success"
    assert len(out["repo_results"]) == 2


def test_backfill_target_repo_name_single_repo() -> None:
    settings = MagicMock()
    settings.is_enabled.return_value = True
    settings.backfill_max_prs_per_run.return_value = 10

    r1 = _repo("p1", "o/r1")
    r2 = _repo("p1", "o/r2")
    resolved = ResolvedPot(pot_id="p1", name="p1", repos=[r1, r2], ready=True)

    pots = MagicMock()
    pots.resolve_pot.return_value = resolved

    repos_called: list[str] = []

    def source_for_repo(name: str) -> MagicMock:
        repos_called.append(name)
        src = MagicMock()
        src.iter_closed_pulls.return_value = []
        return src

    ledger = MagicMock()
    sync = MagicMock()
    sync.last_synced_at = None
    ledger.get_or_create_sync_state.return_value = sync

    context_graph = MagicMock()

    out = backfill_pot_context(
        settings,
        pots,
        source_for_repo,
        ledger,
        context_graph,
        "p1",
        target_repo_name="o/r2",
    )

    assert repos_called == ["o/r2"]
    assert out["status"] == "success"
    assert len(out["repo_results"]) == 1


def test_backfill_unknown_target_repo() -> None:
    settings = MagicMock()
    settings.is_enabled.return_value = True

    resolved = ResolvedPot(
        pot_id="p1",
        name="p1",
        repos=[_repo("p1", "o/r1")],
        ready=True,
    )
    pots = MagicMock()
    pots.resolve_pot.return_value = resolved

    out = backfill_pot_context(
        settings,
        pots,
        lambda _n: MagicMock(),
        MagicMock(),
        MagicMock(),
        "p1",
        target_repo_name="o/missing",
    )

    assert out["status"] == "skipped"
    assert out["reason"] == "repo_not_in_pot"
