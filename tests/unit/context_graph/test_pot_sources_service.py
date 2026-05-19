"""Pot repository mirror into ``context_graph_pot_sources``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.modules.context_graph.pot_sources_service import (
    github_repo_scope_hash,
    github_repo_scope_json,
    repository_for_source,
)
from app.modules.context_graph.wiring import pot_source_row_to_status_source

pytestmark = pytest.mark.unit


def _fake_repo(**overrides: object) -> object:
    fields: dict[str, object] = {
        "provider": "github",
        "provider_host": "github.com",
        "owner": "Acme",
        "repo": "Widget",
        "external_repo_id": "42",
        "remote_url": "https://github.com/Acme/Widget",
        "default_branch": "main",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestScopeHash:
    def test_case_insensitive_owner_repo(self) -> None:
        a = github_repo_scope_hash("github", "github.com", "Acme", "Widget")
        b = github_repo_scope_hash("github", "github.com", "acme", "widget")
        assert a == b

    def test_provider_changes_hash(self) -> None:
        a = github_repo_scope_hash("github", "github.com", "acme", "widget")
        b = github_repo_scope_hash("gitlab", "github.com", "acme", "widget")
        assert a != b

    def test_provider_host_changes_hash(self) -> None:
        a = github_repo_scope_hash("github", "github.com", "acme", "widget")
        b = github_repo_scope_hash("github", "ghe.internal", "acme", "widget")
        assert a != b


class TestScopeJson:
    def test_scope_json_contains_expected_fields(self) -> None:
        scope = json.loads(github_repo_scope_json(_fake_repo()))
        assert scope["owner"] == "Acme"
        assert scope["repo"] == "Widget"
        assert scope["repo_name"] == "Acme/Widget"
        assert scope["provider_host"] == "github.com"
        assert scope["external_repo_id"] == "42"
        assert scope["default_branch"] == "main"

    def test_scope_json_is_deterministic(self) -> None:
        a = github_repo_scope_json(_fake_repo())
        b = github_repo_scope_json(_fake_repo())
        assert a == b


class TestPotSourceRowToStatusSource:
    def _row(self, **overrides: object) -> object:
        base: dict[str, object] = {
            "id": "src_1",
            "pot_id": "pot_1",
            "provider": "github",
            "source_kind": "repository",
            "scope_json": json.dumps(
                {
                    "owner": "acme",
                    "repo": "widget",
                    "repo_name": "acme/widget",
                    "provider_host": "github.com",
                }
            ),
            "sync_enabled": True,
            "sync_mode": None,
            "last_sync_at": None,
            "last_error": None,
            "health_score": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_extracts_provider_host_and_scope_summary_for_repo(self) -> None:
        out = pot_source_row_to_status_source(self._row())  # type: ignore[arg-type]
        assert out.source_id == "src_1"
        assert out.provider_host == "github.com"
        assert out.scope_summary == "acme/widget"

    def test_handles_missing_scope_json(self) -> None:
        out = pot_source_row_to_status_source(self._row(scope_json=None))  # type: ignore[arg-type]
        assert out.provider_host is None
        assert out.scope_summary is None

    def test_summary_for_linear_team(self) -> None:
        out = pot_source_row_to_status_source(  # type: ignore[arg-type]
            self._row(
                provider="linear",
                source_kind="issue_tracker_team",
                scope_json=json.dumps({"team_id": "T1", "team_name": "Core"}),
            )
        )
        assert out.scope_summary == "team:Core"


class _AllQuery:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def filter(self, *_a: object, **_k: object) -> "_AllQuery":
        return self

    def all(self) -> list[object]:
        return self._rows


class _AllDB:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def query(self, _model: object) -> _AllQuery:
        return _AllQuery(self._rows)


class TestRepositoryForSource:
    """Inverse-of-mirror resolution must key on scope_hash, not scope_json."""

    def _source(self, **overrides: object) -> object:
        base: dict[str, object] = {
            "pot_id": "pot_1",
            "provider": "github",
            "source_kind": "repository",
            "scope_hash": github_repo_scope_hash(
                "github", "github.com", "acme", "widget"
            ),
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_matches_repo_row_by_scope_hash_case_insensitively(self) -> None:
        # Repo row stored with different casing than the source's scope —
        # resolution must still match because scope_hash is normalized.
        repo = _fake_repo(owner="Acme", repo="Widget")
        db = _AllDB([_fake_repo(owner="other", repo="thing"), repo])
        out = repository_for_source(db, self._source())  # type: ignore[arg-type]
        assert out is repo

    def test_non_repository_source_returns_none(self) -> None:
        db = _AllDB([_fake_repo()])
        out = repository_for_source(  # type: ignore[arg-type]
            db, self._source(source_kind="issue_tracker_team")
        )
        assert out is None

    def test_no_matching_repo_returns_none(self) -> None:
        db = _AllDB([_fake_repo(owner="someone", repo="else")])
        out = repository_for_source(db, self._source())  # type: ignore[arg-type]
        assert out is None
