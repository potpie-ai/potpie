"""Row-level tenancy for ``UserScopedContextGraphPotResolution``."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.wiring import UserScopedContextGraphPotResolution
from domain.ports.pot_resolution import RepoRef, ResolvedPot

pytestmark = pytest.mark.unit


def _repo_ref() -> RepoRef:
    return RepoRef(
        provider="github",
        provider_host="github.com",
        owner="acme",
        repo="widgets",
    )


def _pot_row(
    *,
    pot_id: str = "pot-1",
    user_id: str = "owner-1",
    archived_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=pot_id,
        user_id=user_id,
        display_name="Team pot",
        archived_at=archived_at,
    )


def _member_row(*, pot_id: str = "pot-1", user_id: str = "member-1") -> SimpleNamespace:
    return SimpleNamespace(pot_id=pot_id, user_id=user_id)


class _FirstQuery:
    def __init__(self, *, result: object | None) -> None:
        self.result = result

    def filter(self, *_args, **_kwargs) -> "_FirstQuery":
        return self

    def first(self) -> object | None:
        return self.result


class _RecordingListQuery:
    """Captures ``.filter(...)`` clauses so tenancy can be asserted."""

    def __init__(self, *, rows: list[tuple[str, ...]]) -> None:
        self._rows = rows
        self.filter_calls: list[tuple[object, ...]] = []

    def join(self, *_args, **_kwargs) -> "_RecordingListQuery":
        return self

    def filter(self, *args, **_kwargs) -> "_RecordingListQuery":
        self.filter_calls.append(args)
        return self

    def all(self) -> list[tuple[str, ...]]:
        return self._rows


def _filter_clauses_target_user(
    filter_calls: list[tuple[object, ...]], user_id: str
) -> bool:
    for call in filter_calls:
        for clause in call:
            if user_id in str(clause):
                return True
            right = getattr(clause, "right", None)
            if right is not None and getattr(right, "value", None) == user_id:
                return True
    return False


def _make_db_for_resolve(results: list[object | None]) -> MagicMock:
    db = MagicMock()
    queries = iter(results)

    def _query(_model) -> _FirstQuery:
        return _FirstQuery(result=next(queries, None))

    db.query.side_effect = _query
    return db


def _make_db_for_find_pots(
    *, member_rows: list[tuple[str, ...]], owner_rows: list[tuple[str, ...]]
) -> tuple[MagicMock, _RecordingListQuery, _RecordingListQuery]:
    member_q = _RecordingListQuery(rows=member_rows)
    owner_q = _RecordingListQuery(rows=owner_rows)
    db = MagicMock()
    db.query.side_effect = [member_q, owner_q]
    return db, member_q, owner_q


def _make_db_for_known_pot_ids(
    *, member_rows: list[tuple[str, ...]], owner_rows: list[tuple[str, ...]]
) -> tuple[MagicMock, _RecordingListQuery, _RecordingListQuery]:
    member_q = _RecordingListQuery(rows=member_rows)
    owner_q = _RecordingListQuery(rows=owner_rows)
    db = MagicMock()
    db.query.side_effect = [member_q, owner_q]
    return db, member_q, owner_q


class TestUserScopedPotResolution:
    def test_actor_scoped_flag_is_true(self) -> None:
        resolver = UserScopedContextGraphPotResolution(MagicMock(), "u1")
        assert resolver.actor_scoped is True

    def test_member_can_resolve_pot(self) -> None:
        pot = _pot_row(user_id="owner-1")
        db = _make_db_for_resolve([pot, _member_row(user_id="member-1")])
        resolved = ResolvedPot(pot_id="pot-1", name="Team pot", repos=[], ready=True)
        with patch(
            "app.modules.context_graph.wiring._resolved_pot_from_context_graph_row",
            return_value=resolved,
        ) as build:
            out = UserScopedContextGraphPotResolution(db, "member-1").resolve_pot(
                "pot-1"
            )
        assert out == resolved
        build.assert_called_once()

    def test_legacy_owner_can_resolve_without_membership_row(self) -> None:
        pot = _pot_row(user_id="owner-1")
        db = _make_db_for_resolve([pot, None, pot])
        resolved = ResolvedPot(pot_id="pot-1", name="Team pot", repos=[], ready=True)
        with patch(
            "app.modules.context_graph.wiring._resolved_pot_from_context_graph_row",
            return_value=resolved,
        ) as build:
            out = UserScopedContextGraphPotResolution(db, "owner-1").resolve_pot(
                "pot-1"
            )
        assert out == resolved
        build.assert_called_once()

    def test_archived_pot_is_hidden(self) -> None:
        pot = _pot_row(archived_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
        db = _make_db_for_resolve([pot])
        with patch(
            "app.modules.context_graph.wiring._resolved_pot_from_context_graph_row",
        ) as build:
            out = UserScopedContextGraphPotResolution(db, "owner-1").resolve_pot(
                "pot-1"
            )
        assert out is None
        build.assert_not_called()

    def test_non_member_non_owner_denied(self) -> None:
        pot = _pot_row(user_id="owner-1")
        db = _make_db_for_resolve([pot, None, pot])
        with patch(
            "app.modules.context_graph.wiring._resolved_pot_from_context_graph_row",
        ) as build:
            out = UserScopedContextGraphPotResolution(db, "stranger").resolve_pot(
                "pot-1"
            )
        assert out is None
        build.assert_not_called()

    def test_find_pots_for_repo_merges_member_and_owner_queries(self) -> None:
        db, _member_q, _owner_q = _make_db_for_find_pots(
            member_rows=[("pot-member",)],
            owner_rows=[("pot-owner",)],
        )
        out = UserScopedContextGraphPotResolution(db, "user-1").find_pots_for_repo(
            _repo_ref()
        )
        assert out == ["pot-member", "pot-owner"]

    def test_find_pots_for_repo_deduplicates(self) -> None:
        db, _, _ = _make_db_for_find_pots(
            member_rows=[("pot-1",)],
            owner_rows=[("pot-1",)],
        )
        out = UserScopedContextGraphPotResolution(db, "user-1").find_pots_for_repo(
            _repo_ref()
        )
        assert out == ["pot-1"]

    def test_find_pots_for_repo_scopes_both_queries_to_resolver_user(self) -> None:
        db, member_q, owner_q = _make_db_for_find_pots(member_rows=[], owner_rows=[])
        UserScopedContextGraphPotResolution(db, "user-42").find_pots_for_repo(
            _repo_ref()
        )
        assert _filter_clauses_target_user(member_q.filter_calls, "user-42")
        assert _filter_clauses_target_user(owner_q.filter_calls, "user-42")

    def test_find_pots_for_repo_returns_empty_when_user_has_no_access(self) -> None:
        db, _, _ = _make_db_for_find_pots(member_rows=[], owner_rows=[])
        out = UserScopedContextGraphPotResolution(db, "stranger").find_pots_for_repo(
            _repo_ref()
        )
        assert out == []

    def test_known_pot_ids_scopes_both_queries_to_resolver_user(self) -> None:
        db, member_q, owner_q = _make_db_for_known_pot_ids(
            member_rows=[("pot-m",)],
            owner_rows=[("pot-o",)],
        )
        out = UserScopedContextGraphPotResolution(db, "user-99").known_pot_ids()
        assert out == ["pot-m", "pot-o"]
        assert _filter_clauses_target_user(member_q.filter_calls, "user-99")
        assert _filter_clauses_target_user(owner_q.filter_calls, "user-99")
