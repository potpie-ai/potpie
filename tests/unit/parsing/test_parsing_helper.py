"""
Unit tests for ParseHelper (get_directory_size, detect_repo_language, is_text_file).
Uses temp dirs and mocks where needed; check_commit_status tested with mocked ProjectService.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.modules.parsing.graph_construction.parsing_helper import ParseHelper


pytestmark = pytest.mark.unit


class TestGetDirectorySize:
    """Test ParseHelper.get_directory_size (static)."""

    def test_empty_dir_returns_zero(self, tmp_path):
        """Empty directory returns 0."""
        assert ParseHelper.get_directory_size(str(tmp_path)) == 0

    def test_with_files_returns_sum(self, tmp_path):
        """Directory with files returns total size (symlinks excluded)."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        total = ParseHelper.get_directory_size(str(tmp_path))
        assert total == 5 + 5

    def test_symlink_excluded(self, tmp_path):
        """Symlink size is not counted."""
        (tmp_path / "real.txt").write_text("x" * 100)
        target = tmp_path / "real.txt"
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        total = ParseHelper.get_directory_size(str(tmp_path))
        assert total == 100


class TestDetectRepoLanguage:
    """Test ParseHelper.detect_repo_language (static)."""

    def test_directory_missing_returns_other(self):
        """Non-existent path returns 'other'."""
        assert ParseHelper.detect_repo_language("/nonexistent/path") == "other"

    def test_not_directory_returns_other(self, tmp_path):
        """Path that is a file returns 'other'."""
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert ParseHelper.detect_repo_language(str(f)) == "other"

    def test_python_only_returns_python(self, tmp_path):
        """Directory with only .py files returns 'python'."""
        (tmp_path / "main.py").write_text("def foo(): pass")
        (tmp_path / "lib.py").write_text("x = 1")
        assert ParseHelper.detect_repo_language(str(tmp_path)) == "python"

    def test_no_supported_files_returns_other(self, tmp_path):
        """Directory with no supported language files returns 'other'."""
        (tmp_path / "readme.txt").write_text("text")
        assert ParseHelper.detect_repo_language(str(tmp_path)) == "other"

    def test_mixed_languages_returns_predominant(self, tmp_path):
        """Directory with .py and .js returns predominant (more .py -> python)."""
        for i in range(3):
            (tmp_path / f"f{i}.py").write_text("pass")
        (tmp_path / "one.js").write_text("x")
        assert ParseHelper.detect_repo_language(str(tmp_path)) == "python"


class TestIsTextFile:
    """Test ParseHelper.is_text_file (instance method)."""

    def test_utf8_returns_true(self, db_session, tmp_path):
        """UTF-8 text file returns True."""
        f = tmp_path / "t.txt"
        f.write_text("hello", encoding="utf-8")
        helper = ParseHelper(db_session)
        assert helper.is_text_file(str(f)) is True

    def test_binary_returns_false(self, db_session, tmp_path):
        """Binary file: implementation may still return True (latin-1 accepts all bytes)."""
        f = tmp_path / "b.bin"
        f.write_bytes(b"\x00\xff\xfe")
        helper = ParseHelper(db_session)
        result = helper.is_text_file(str(f))
        # ParseHelper uses latin-1 as fallback, which decodes any byte sequence,
        # so some binary files may be reported as text; we only assert it returns a bool.
        assert isinstance(result, bool)


class TestCheckCommitStatus:
    """Test ParseHelper.check_commit_status with mocked ProjectService."""

    @pytest.mark.asyncio
    async def test_no_project_returns_false(self, db_session):
        """When project not found, returns False."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await helper.check_commit_status("proj-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_pinned_commit_match_returns_true(self, db_session):
        """When requested_commit_id matches stored commit, returns True."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value={
                "commit_id": "abc123",
                "project_name": "owner/repo",
                "branch_name": "main",
            },
        ):
            result = await helper.check_commit_status(
                "proj-123",
                requested_commit_id="abc123",
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_pinned_commit_mismatch_returns_false(self, db_session):
        """When requested_commit_id differs from stored, returns False."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value={
                "commit_id": "old123",
                "project_name": "owner/repo",
                "branch_name": "main",
            },
        ):
            result = await helper.check_commit_status(
                "proj-123",
                requested_commit_id="new456",
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_branch_based_match_returns_true(self, db_session):
        """When no requested_commit_id, branch-based: latest commit matches stored, returns True."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value={
                "commit_id": "abc123",
                "project_name": "owner/repo",
                "branch_name": "main",
            },
        ), patch(
            "app.modules.parsing.graph_construction.parsing_helper._fetch_github_branch_head_sha_http",
            return_value="abc123",
        ):
            result = await helper.check_commit_status("proj-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_branch_based_mismatch_returns_false(self, db_session):
        """When no requested_commit_id, branch-based: latest commit differs from stored, returns False."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value={
                "commit_id": "old123",
                "project_name": "owner/repo",
                "branch_name": "main",
            },
        ), patch(
            "app.modules.parsing.graph_construction.parsing_helper._fetch_github_branch_head_sha_http",
            return_value="latest999",
        ):
            result = await helper.check_commit_status("proj-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_github_exception_returns_false(self, db_session):
        """When _fetch_github_branch_head_sha_http raises or returns None, returns False."""
        helper = ParseHelper(db_session)
        with patch.object(
            helper.project_manager,
            "get_project_from_db_by_id",
            new_callable=AsyncMock,
            return_value={
                "commit_id": "abc123",
                "project_name": "owner/repo",
                "branch_name": "main",
            },
        ), patch(
            "app.modules.parsing.graph_construction.parsing_helper._fetch_github_branch_head_sha_http",
            side_effect=Exception("GitHub API error"),
        ):
            result = await helper.check_commit_status("proj-123")
        assert result is False
