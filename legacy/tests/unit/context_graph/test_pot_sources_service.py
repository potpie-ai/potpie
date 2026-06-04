"""Pot repository mirror into ``context_graph_pot_sources``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.modules.context_graph.pot_sources_service import (
    github_repo_scope_hash,
    github_repo_scope_json,
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
