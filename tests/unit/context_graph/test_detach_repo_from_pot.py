"""``detach_repo_from_pot`` use case — DB + sandbox cleanup wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.detach_repo_from_pot import (
    DetachRepoResult,
    UnknownPotError,
    UnknownRepositoryError,
    detach_repo_from_pot,
)

pytestmark = pytest.mark.unit


def _fake_pot(*, pot_id: str = "pot-1") -> SimpleNamespace:
    return SimpleNamespace(id=pot_id, primary_repo_name="acme/widgets")


def _fake_repo_row(
    *,
    rid: str = "r-1",
    pot_id: str = "pot-1",
    owner: str = "acme",
    repo: str = "widgets",
    added_by_user_id: str = "user-7",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=rid,
        pot_id=pot_id,
        owner=owner,
        repo=repo,
        added_by_user_id=added_by_user_id,
    )


class _Query:
    def __init__(self, *, result: object | None) -> None:
        self.result = result

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args):
        return self

    def first(self):
        return self.result


def _make_db(*, query_results: list[object | None]) -> MagicMock:
    db = MagicMock()
    queries = iter(query_results)

    def _query(_model):
        return _Query(result=next(queries, None))

    db.query.side_effect = _query
    db.delete = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    return db


class TestDetachRepoFromPot:
    def test_unknown_pot_raises(self) -> None:
        db = _make_db(query_results=[None])
        with pytest.raises(UnknownPotError):
            detach_repo_from_pot(db, pot_id="missing", repository_id="r-1")

    def test_unknown_repository_raises(self) -> None:
        db = _make_db(query_results=[_fake_pot(), None])
        with pytest.raises(UnknownRepositoryError):
            detach_repo_from_pot(db, pot_id="pot-1", repository_id="ghost")

    def test_happy_path_drops_row_and_dispatches_sandbox_cleanup(self) -> None:
        pot = _fake_pot()
        row = _fake_repo_row()
        # 4 queries: pot lookup, repo lookup, _recompute (pot), _recompute (first).
        db = _make_db(query_results=[pot, row, pot, None])
        with patch(
            "app.modules.context_graph.detach_repo_from_pot.unmirror_repository_from_sources",
        ) as unmirror, patch(
            "app.modules.context_graph.detach_repo_from_pot._dispatch_sandbox_detach"
        ) as dispatch:
            result = detach_repo_from_pot(
                db, pot_id="pot-1", repository_id="r-1"
            )
        assert isinstance(result, DetachRepoResult)
        assert result.repository_id == "r-1"
        assert result.owner == "acme"
        assert result.repo == "widgets"
        unmirror.assert_called_once_with(db, row)
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()
        # Sandbox cleanup is dispatched with the canonical key shape.
        dispatch.assert_called_once_with(
            user_id="user-7", pot_id="pot-1", repo="acme/widgets"
        )

    def test_dispatch_sandbox_detach_swallows_missing_user(self) -> None:
        from app.modules.context_graph.detach_repo_from_pot import (
            _dispatch_sandbox_detach,
        )

        # No user_id -> early return, no thread, no import.
        _dispatch_sandbox_detach(user_id=None, pot_id="pot-1", repo="acme/widgets")
