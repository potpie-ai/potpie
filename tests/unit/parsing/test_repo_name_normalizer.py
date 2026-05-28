"""
Unit tests for repo_name_normalizer (normalize_repo_name, get_actual_repo_name_for_lookup).
"""

import pytest

from app.modules.parsing.utils.repo_name_normalizer import (
    normalize_repo_name,
    get_actual_repo_name_for_lookup,
)


pytestmark = pytest.mark.unit


class TestNormalizeRepoName:
    """Test normalize_repo_name."""

    def test_none_returns_as_is(self):
        """None returns as-is."""
        assert normalize_repo_name(None) is None

    def test_empty_returns_as_is(self):
        """Empty string returns as-is."""
        assert normalize_repo_name("") == ""

    def test_no_slash_returns_as_is(self):
        """Repo name without slash returns as-is."""
        assert normalize_repo_name("noproject") == "noproject"

    def test_github_style_returns_as_is(self):
        """owner/repo with provider github returns as-is."""
        assert normalize_repo_name("owner/repo", provider_type="github") == "owner/repo"

    def test_gitbucket_root_with_username(self, monkeypatch):
        """GitBucket root/repo with GITBUCKET_USERNAME set normalizes to username/repo."""
        monkeypatch.setenv("GITBUCKET_USERNAME", "alice")
        assert normalize_repo_name("root/repo", provider_type="gitbucket") == "alice/repo"

    def test_gitbucket_root_no_username(self, monkeypatch):
        """GitBucket root/repo without GITBUCKET_USERNAME returns as-is."""
        monkeypatch.delenv("GITBUCKET_USERNAME", raising=False)
        assert normalize_repo_name("root/repo", provider_type="gitbucket") == "root/repo"

    def test_gitbucket_from_env(self, monkeypatch):
        """Provider from CODE_PROVIDER env when not passed."""
        monkeypatch.setenv("CODE_PROVIDER", "github")
        assert normalize_repo_name("owner/repo") == "owner/repo"


class TestGetActualRepoNameForLookup:
    """Test get_actual_repo_name_for_lookup."""

    def test_github_returns_as_is(self):
        """For github, returns repo_name as-is."""
        assert get_actual_repo_name_for_lookup("owner/repo", "github") == "owner/repo"

    def test_gitbucket_username_format(self, monkeypatch):
        """GitBucket: repo already username/repo with GITBUCKET_USERNAME set returns as-is."""
        monkeypatch.setenv("GITBUCKET_USERNAME", "alice")
        assert get_actual_repo_name_for_lookup("alice/repo", "gitbucket") == "alice/repo"

    def test_gitbucket_root_format(self, monkeypatch):
        """GitBucket: root/repo returns as-is."""
        assert get_actual_repo_name_for_lookup("root/repo", "gitbucket") == "root/repo"

    def test_gitbucket_fallback_to_root(self, monkeypatch):
        """GitBucket: username set but repo not username-prefixed converts to root/repo."""
        monkeypatch.setenv("GITBUCKET_USERNAME", "alice")
        # repo_name "other/repo" -> root/repo for API
        result = get_actual_repo_name_for_lookup("other/repo", "gitbucket")
        assert result == "root/repo"

    def test_none_returns_as_is(self):
        """None returns as-is."""
        assert get_actual_repo_name_for_lookup(None) is None

    def test_empty_returns_as_is(self):
        """Empty string returns as-is."""
        assert get_actual_repo_name_for_lookup("") == ""
