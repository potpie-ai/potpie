"""
Unit tests for parsing schema (ParsingRequest, RepoDetails, ParsingStatusRequest).
"""

import pytest
from pydantic import ValidationError

from app.modules.parsing.graph_construction.parsing_schema import (
    ParsingRequest,
    ParsingResponse,
    RepoDetails,
    ParsingStatusRequest,
)


pytestmark = pytest.mark.unit


class TestParsingRequestRepositoryIdentifier:
    """Test ParsingRequest with the new repository_identifier field."""

    def test_remote_repo_via_identifier(self):
        """repository_identifier with remote repo resolves correctly."""
        req = ParsingRequest(repository_identifier="owner/repo")
        assert req.repository_identifier == "owner/repo"
        assert req.repo_name == "owner/repo"
        assert req.repo_path is None

    def test_local_path_via_identifier(self):
        """repository_identifier with absolute path resolves correctly."""
        req = ParsingRequest(repository_identifier="/some/local/path")
        assert req.repository_identifier == "/some/local/path"
        assert req.repo_path == "/some/local/path"
        assert req.repo_name == "path"

    def test_home_relative_path_via_identifier(self):
        """repository_identifier with ~ path resolves correctly."""
        import os
        req = ParsingRequest(repository_identifier="~/repos/myproject")
        assert req.repo_path == os.path.expanduser("~/repos/myproject")
        assert req.repo_name == "myproject"

    def test_dot_relative_path_via_identifier(self):
        """repository_identifier with ./ path resolves correctly."""
        req = ParsingRequest(repository_identifier="./myproject")
        assert req.repo_name == "myproject"
        assert req.repo_path is not None

    def test_identifier_with_branch_and_commit(self):
        """repository_identifier works alongside branch_name and commit_id."""
        req = ParsingRequest(
            repository_identifier="owner/repo",
            branch_name="main",
            commit_id="abc123",
        )
        assert req.repo_name == "owner/repo"
        assert req.branch_name == "main"
        assert req.commit_id == "abc123"

    def test_identifier_takes_precedence_over_legacy(self):
        """repository_identifier takes precedence over repo_name/repo_path."""
        req = ParsingRequest(
            repository_identifier="owner/repo",
            repo_name="other/repo",
            repo_path="/some/path",
        )
        # repository_identifier should win
        assert req.repo_name == "owner/repo"
        assert req.repo_path is None


class TestParsingRequestBackwardCompat:
    """Test backward compatibility with deprecated repo_name/repo_path fields."""

    def test_valid_with_repo_name_only(self):
        """ParsingRequest with repo_name only is valid."""
        req = ParsingRequest(repo_name="owner/repo")
        assert req.repo_name == "owner/repo"
        assert req.repo_path is None

    def test_valid_with_repo_path_only(self):
        """ParsingRequest with repo_path only is valid."""
        req = ParsingRequest(repo_path="/some/path")
        assert req.repo_path == "/some/path"
        assert req.repo_name is None

    def test_valid_with_both_repo_name_and_path(self):
        """ParsingRequest with both repo_name and repo_path is valid."""
        req = ParsingRequest(repo_name="owner/repo", repo_path="/tmp/repo")
        assert req.repo_name == "owner/repo"
        assert req.repo_path == "/tmp/repo"

    def test_invalid_when_all_missing(self):
        """ParsingRequest raises ValueError when no identifier provided."""
        with pytest.raises(
            ValueError,
            match="Either repository_identifier, repo_name, or repo_path must be provided",
        ):
            ParsingRequest()

    def test_invalid_with_empty_strings_both_none(self):
        """repo_name='', repo_path=None triggers validation (both falsy)."""
        with pytest.raises(
            ValueError,
            match="Either repository_identifier, repo_name, or repo_path must be provided",
        ):
            ParsingRequest(repo_name="", repo_path=None)

    def test_optional_branch_and_commit(self):
        """branch_name and commit_id are optional."""
        req = ParsingRequest(repo_name="a/b", branch_name="main", commit_id="abc123")
        assert req.branch_name == "main"
        assert req.commit_id == "abc123"

    def test_repo_name_auto_detected_as_path(self):
        """repo_name that looks like a path is auto-resolved to repo_path."""
        req = ParsingRequest(repo_name="/absolute/path/to/myrepo")
        assert req.repo_path == "/absolute/path/to/myrepo"
        assert req.repo_name == "myrepo"


class TestRepoDetails:
    """Test RepoDetails model."""

    def test_construction_with_required_only(self):
        """RepoDetails with required repo_name and branch_name."""
        details = RepoDetails(repo_name="owner/repo", branch_name="main")
        assert details.repo_name == "owner/repo"
        assert details.branch_name == "main"
        assert details.repo_path is None
        assert details.commit_id is None
        assert details.is_local is False

    def test_construction_with_optional(self):
        """RepoDetails with optional repo_path and commit_id."""
        details = RepoDetails(
            repo_name="owner/repo",
            branch_name="main",
            repo_path="/local/repo",
            commit_id="abc123",
        )
        assert details.repo_path == "/local/repo"
        assert details.commit_id == "abc123"

    def test_is_local_flag(self):
        """RepoDetails is_local flag."""
        details = RepoDetails(
            repo_name="myrepo",
            branch_name="main",
            repo_path="/local/repo",
            is_local=True,
        )
        assert details.is_local is True

    def test_is_local_default_false(self):
        """RepoDetails is_local defaults to False."""
        details = RepoDetails(repo_name="owner/repo", branch_name="main")
        assert details.is_local is False


class TestParsingStatusRequest:
    """Test ParsingStatusRequest model."""

    def test_repo_name_required(self):
        """ParsingStatusRequest requires repo_name."""
        req = ParsingStatusRequest(repo_name="owner/repo")
        assert req.repo_name == "owner/repo"
        assert req.commit_id is None
        assert req.branch_name is None

    def test_optional_commit_and_branch(self):
        """commit_id and branch_name are optional."""
        req = ParsingStatusRequest(
            repo_name="a/b",
            commit_id="abc",
            branch_name="main",
        )
        assert req.commit_id == "abc"
        assert req.branch_name == "main"


class TestParsingResponse:
    """Test ParsingResponse model."""

    def test_construction(self):
        """ParsingResponse has message, status, project_id."""
        resp = ParsingResponse(
            message="Done",
            status="ready",
            project_id="proj-123",
        )
        assert resp.message == "Done"
        assert resp.status == "ready"
        assert resp.project_id == "proj-123"


class TestInputEdgeCases:
    """Test edge cases for input validation (Part 4.2 of plan)."""

    def test_repo_name_whitespace_only(self):
        """repo_name with only whitespace is accepted at schema level (no strip)."""
        # NOTE: Pydantic does NOT strip whitespace by default, so "   " is truthy.
        # The validation "both missing" only checks falsy values.
        # This documents actual behavior - whitespace-only passes schema validation.
        req = ParsingRequest(repo_name="   ", repo_path=None)
        assert req.repo_name == "   "
        # Downstream code should handle this edge case

    def test_repo_name_no_slash(self):
        """repo_name without slash (e.g. 'myrepo') is valid at schema level."""
        req = ParsingRequest(repo_name="myrepo")
        assert req.repo_name == "myrepo"

    def test_repo_name_multiple_slashes(self):
        """repo_name with multiple slashes (e.g. 'org/repo/sub') is valid at schema level."""
        req = ParsingRequest(repo_name="org/repo/subpath")
        assert req.repo_name == "org/repo/subpath"

    def test_repo_name_leading_trailing_spaces(self):
        """repo_name with leading/trailing spaces."""
        req = ParsingRequest(repo_name="  owner/repo  ")
        # Pydantic may or may not strip - document actual behavior
        assert "owner/repo" in req.repo_name

    def test_empty_commit_id_string(self):
        """Empty commit_id string should be accepted or converted to None."""
        req = ParsingRequest(repo_name="owner/repo", commit_id="")
        # Empty string is valid at schema level - may be handled downstream
        assert req.commit_id == "" or req.commit_id is None

    def test_empty_branch_name_string(self):
        """Empty branch_name string should be accepted."""
        req = ParsingRequest(repo_name="owner/repo", branch_name="")
        assert req.branch_name == "" or req.branch_name is None

    def test_very_long_repo_name(self):
        """Very long repo_name should be accepted at schema level."""
        long_name = "owner/" + "x" * 500
        req = ParsingRequest(repo_name=long_name)
        assert req.repo_name == long_name

    def test_special_characters_in_repo_name(self):
        """repo_name with special characters (dashes, underscores, dots)."""
        req = ParsingRequest(repo_name="my-org/my_repo.name")
        assert req.repo_name == "my-org/my_repo.name"

    def test_unicode_in_repo_name(self):
        """repo_name with unicode characters."""
        req = ParsingRequest(repo_name="owner/репо")
        assert req.repo_name == "owner/репо"

    def test_model_dump_includes_repository_identifier(self):
        """model_dump() includes repository_identifier for Celery serialization."""
        req = ParsingRequest(repository_identifier="owner/repo", branch_name="main")
        dumped = req.model_dump()
        assert "repository_identifier" in dumped
        assert dumped["repository_identifier"] == "owner/repo"
        assert dumped["repo_name"] == "owner/repo"

    def test_model_dump_roundtrip(self):
        """ParsingRequest can be reconstructed from model_dump()."""
        req = ParsingRequest(repository_identifier="owner/repo", branch_name="main")
        dumped = req.model_dump()
        reconstructed = ParsingRequest(**dumped)
        assert reconstructed.repo_name == req.repo_name
        assert reconstructed.repo_path == req.repo_path
        assert reconstructed.branch_name == req.branch_name
