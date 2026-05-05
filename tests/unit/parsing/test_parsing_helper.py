"""Unit tests for parsing_helper (e.g. _fetch_github_branch_head_sha_http)."""
from unittest.mock import MagicMock, patch

import pytest

from app.modules.parsing.graph_construction.parsing_helper import (
    _fetch_github_branch_head_sha_http,
    ParsingFailedError,
    ParsingServiceError,
)


pytestmark = pytest.mark.unit


class TestFetchGithubBranchHeadShaHttp:
    @patch("app.modules.parsing.graph_construction.parsing_helper.urllib.request.urlopen")
    @patch("app.modules.parsing.graph_construction.parsing_helper.os.getenv")
    def test_returns_sha_when_success(self, mock_getenv, mock_urlopen):
        def getenv(k, d=""):
            return {"GH_TOKEN_LIST": "token", "CODE_PROVIDER_TOKEN": ""}.get(k, d or "")
        mock_getenv.side_effect = getenv
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"commit": {"sha": "abc123"}}'
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_resp)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = ctx
        result = _fetch_github_branch_head_sha_http("owner/repo", "main")
        assert result == "abc123"

    @patch("app.modules.parsing.graph_construction.parsing_helper.urllib.request.urlopen")
    @patch("app.modules.parsing.graph_construction.parsing_helper.os.getenv")
    def test_returns_none_on_exception(self, mock_getenv, mock_urlopen):
        mock_getenv.return_value = ""
        mock_urlopen.side_effect = Exception("network error")
        result = _fetch_github_branch_head_sha_http("owner/repo", "main")
        assert result is None

    @patch("app.modules.parsing.graph_construction.parsing_helper.urllib.request.urlopen")
    @patch("app.modules.parsing.graph_construction.parsing_helper.os.getenv")
    def test_returns_none_when_no_commit_in_response(self, mock_getenv, mock_urlopen):
        mock_getenv.return_value = ""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{}'
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_resp)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = ctx
        result = _fetch_github_branch_head_sha_http("owner/repo", "main")
        assert result is None


class TestParsingExceptions:
    def test_parsing_service_error(self):
        err = ParsingServiceError("parse failed")
        assert str(err) == "parse failed"

    def test_parsing_failed_error_inherits(self):
        err = ParsingFailedError("failed")
        assert isinstance(err, ParsingServiceError)


# Phase 5 retired the SANDBOX_PARSING_ENABLED flag and the
# _clone_via_sandbox / clone_or_copy_repository fallback chain.
# parse_directory now goes through ProjectSandbox unconditionally —
# see tests/unit/parsing/test_sandbox_parse_flow.py for the new
# coverage.
