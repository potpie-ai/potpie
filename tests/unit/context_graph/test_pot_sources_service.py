"""Pot repository mirror into ``context_graph_pot_sources``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.modules.context_graph.pot_sources_service import (
    github_repo_scope_hash,
    github_repo_scope_json,
)

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
