"""``attach_repo_to_pot`` use case — idempotency, dedup, error mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.attach_repo_to_pot import (
    AttachRepoResult,
    UnknownPotError,
    attach_repo_to_pot,
)

pytestmark = pytest.mark.unit


def _fake_pot(*, pot_id: str = "pot-1", primary_repo_name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=pot_id, primary_repo_name=primary_repo_name)


def _fake_repo_row(
    *,
    rid: str = "r-1",
    pot_id: str = "pot-1",
    owner: str = "acme",
    repo: str = "widgets",
    provider: str = "github",
    provider_host: str = "github.com",
    default_branch: str = "main",
    remote_url: str = "https://github.com/acme/widgets",
    external_repo_id: str = "42",
    added_by_user_id: str = "user-7",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=rid,
        pot_id=pot_id,
        owner=owner,
        repo=repo,
        provider=provider,
        provider_host=provider_host,
        default_branch=default_branch,
        remote_url=remote_url,
        external_repo_id=external_repo_id,
        added_by_user_id=added_by_user_id,
    )


def _fake_source(sid: str = "src-1") -> SimpleNamespace:
    return SimpleNamespace(id=sid)


class _Query:
    """Recording mock for ``db.query(Model).filter(...).first()``."""

    def __init__(self, *, result: object | None) -> None:
        self.result = result
        self.filter_calls: list[tuple] = []

    def filter(self, *args, **_kwargs) -> "_Query":
        self.filter_calls.append(args)
        return self

    def order_by(self, *_args) -> "_Query":
        return self

    def first(self) -> object | None:
        return self.result


def _make_db(*, query_results: list[object | None]) -> MagicMock:
    """Hand back canned ``.first()`` results in the order queries arrive."""
    db = MagicMock()
    queries = iter(query_results)

    def _query(_model):
        return _Query(result=next(queries, None))

    db.query.side_effect = _query
    db.add = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock()
    db.delete = MagicMock()
    return db


class TestAttachRepoToPot:
    def test_unknown_pot_raises(self) -> None:
        db = _make_db(query_results=[None])
        with pytest.raises(UnknownPotError):
            attach_repo_to_pot(
                db,
                pot_id="missing",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_empty_owner_or_repo_raises_value_error(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_first_attach_creates_row_and_emits_event(self) -> None:
        pot = _fake_pot(primary_repo_name=None)
        db = _make_db(query_results=[pot, None])
        source = _fake_source("src-NEW")
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=source,
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value="evt-bootstrap-1",
        ) as emit:
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id="42",
                remote_url="https://github.com/acme/widgets",
                default_branch="main",
                submitted_by_user_id="user-7",
            )
        assert isinstance(result, AttachRepoResult)
        assert result.already_attached is False
        assert result.bootstrap_event_id == "evt-bootstrap-1"
        assert result.source_id == "src-NEW"
        # Row was actually added + committed.
        assert db.add.called
        assert db.commit.called
        # Primary repo back-filled when previously unset.
        assert pot.primary_repo_name == "acme/widgets"
        # Bootstrap event called exactly once.
        assert emit.call_count == 1

    def test_existing_repo_is_idempotent_and_skips_event(self) -> None:
        pot = _fake_pot(primary_repo_name="acme/widgets")
        existing = _fake_repo_row()
        db = _make_db(query_results=[pot, existing])
        source = _fake_source("src-DUP")
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=source,
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
        ) as emit:
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id="42",
                remote_url="https://github.com/acme/widgets",
                default_branch="main",
                submitted_by_user_id="user-7",
            )
        assert result.already_attached is True
        assert result.bootstrap_event_id is None
        assert result.repository_id == existing.id
        assert result.source_id == "src-DUP"
        # No new row added on re-attach.
        assert not db.add.called
        # Bootstrap event NOT re-emitted on idempotent re-attach.
        assert emit.call_count == 0

    def test_first_attach_preserves_existing_primary_repo_name(self) -> None:
        pot = _fake_pot(primary_repo_name="other/repo")
        db = _make_db(query_results=[pot, None])
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=_fake_source(),
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value="evt-2",
        ):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch="main",
                submitted_by_user_id="u",
            )
        # Pre-existing primary_repo_name MUST NOT be overwritten.
        assert pot.primary_repo_name == "other/repo"

    def test_inputs_are_trimmed_before_persistence(self) -> None:
        pot = _fake_pot(primary_repo_name=None)
        db = _make_db(query_results=[pot, None])
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=_fake_source(),
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value=None,
        ):
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="  github  ",
                provider_host="  github.com  ",
                owner="  acme  ",
                repo="  widgets  ",
                external_repo_id="  42  ",
                remote_url="  https://github.com/acme/widgets  ",
                default_branch="  main  ",
                submitted_by_user_id="u",
            )
        row = result.repository
        assert row.owner == "acme"
        assert row.repo == "widgets"
        assert row.provider == "github"
        assert row.provider_host == "github.com"
        assert row.external_repo_id == "42"
        assert row.remote_url == "https://github.com/acme/widgets"
        assert row.default_branch == "main"
