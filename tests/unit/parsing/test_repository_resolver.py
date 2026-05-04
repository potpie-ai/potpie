"""
Unit tests for RepositoryResolver.
"""

import os

import pytest

from app.modules.parsing.utils.repository_resolver import (
    RepositoryResolver,
    RepositoryType,
    ResolvedRepository,
)


pytestmark = pytest.mark.unit


class TestLooksLikePath:
    """Test RepositoryResolver.looks_like_path."""

    def test_absolute_unix_path(self):
        assert RepositoryResolver.looks_like_path("/home/user/repo") is True

    def test_home_relative_path(self):
        assert RepositoryResolver.looks_like_path("~/repos/myproject") is True

    def test_dot_relative_path(self):
        assert RepositoryResolver.looks_like_path("./myproject") is True

    def test_parent_relative_path(self):
        assert RepositoryResolver.looks_like_path("../myproject") is True

    def test_remote_repo_identifier(self):
        assert RepositoryResolver.looks_like_path("owner/repo") is False

    def test_simple_name(self):
        assert RepositoryResolver.looks_like_path("myrepo") is False

    def test_empty_string(self):
        assert RepositoryResolver.looks_like_path("") is False

    def test_none_is_false(self):
        # looks_like_path should handle None gracefully
        assert RepositoryResolver.looks_like_path(None) is False


class TestClassify:
    """Test RepositoryResolver.classify."""

    def test_remote_repo(self):
        result = RepositoryResolver.classify("owner/repo")
        assert result.is_local is False
        assert result.repo_name == "owner/repo"
        assert result.repo_path is None
        assert result.repository_identifier == "owner/repo"

    def test_absolute_local_path(self):
        result = RepositoryResolver.classify("/home/user/myrepo")
        assert result.is_local is True
        assert result.repo_name == "myrepo"
        assert result.repo_path == "/home/user/myrepo"
        assert result.repository_identifier == "/home/user/myrepo"

    def test_home_relative_path(self):
        result = RepositoryResolver.classify("~/repos/myproject")
        assert result.is_local is True
        assert result.repo_name == "myproject"
        # repo_path should be expanded
        assert result.repo_path == os.path.expanduser("~/repos/myproject")
        assert result.repository_identifier == "~/repos/myproject"

    def test_dot_relative_path(self):
        result = RepositoryResolver.classify("./myproject")
        assert result.is_local is True
        assert result.repo_name == "myproject"

    def test_parent_relative_path(self):
        result = RepositoryResolver.classify("../myproject")
        assert result.is_local is True
        assert result.repo_name == "myproject"

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            RepositoryResolver.classify("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            RepositoryResolver.classify("   ")

    def test_strips_whitespace(self):
        result = RepositoryResolver.classify("  owner/repo  ")
        assert result.repo_name == "owner/repo"
        assert result.is_local is False

    def test_remote_with_dots_in_name(self):
        result = RepositoryResolver.classify("my-org/my_repo.name")
        assert result.is_local is False
        assert result.repo_name == "my-org/my_repo.name"

    def test_remote_with_multiple_slashes(self):
        """GitLab-style paths with multiple levels."""
        result = RepositoryResolver.classify("org/group/repo")
        assert result.is_local is False
        assert result.repo_name == "org/group/repo"


class TestExtractRepoNameFromPath:
    """Test RepositoryResolver.extract_repo_name_from_path."""

    def test_simple_path(self):
        assert RepositoryResolver.extract_repo_name_from_path("/path/to/myrepo") == "myrepo"

    def test_trailing_slash(self):
        assert RepositoryResolver.extract_repo_name_from_path("/path/to/myrepo/") == "myrepo"

    def test_single_component(self):
        assert RepositoryResolver.extract_repo_name_from_path("myrepo") == "myrepo"

    def test_home_path(self):
        assert RepositoryResolver.extract_repo_name_from_path("/home/user/project") == "project"

    def test_dot_directory(self):
        assert RepositoryResolver.extract_repo_name_from_path("/home/user/.hidden") == ".hidden"


class TestResolvedRepositoryNamedTuple:
    """Test ResolvedRepository is a proper NamedTuple."""

    def test_fields(self):
        resolved = ResolvedRepository(
            repository_identifier="owner/repo",
            repo_name="owner/repo",
            repo_path=None,
            is_local=False,
        )
        assert resolved.repository_identifier == "owner/repo"
        assert resolved.repo_name == "owner/repo"
        assert resolved.repo_path is None
        assert resolved.is_local is False

    def test_unpacking(self):
        resolved = ResolvedRepository(
            repository_identifier="/tmp/repo",
            repo_name="repo",
            repo_path="/tmp/repo",
            is_local=True,
        )
        ident, name, path, is_local = resolved
        assert ident == "/tmp/repo"
        assert name == "repo"
        assert path == "/tmp/repo"
        assert is_local is True
