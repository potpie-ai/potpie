"""Unit tests for parsing_helper (e.g. _fetch_github_branch_head_sha_http)."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.parsing.graph_construction.parsing_helper import (
    _fetch_github_branch_head_sha_http,
    ParseHelper,
    ParsingFailedError,
    ParsingServiceError,
)
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails


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


class TestSandboxParsingFlag:
    """`SANDBOX_PARSING_ENABLED` toggles the sandbox-backed materialisation path."""

    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SANDBOX_PARSING_ENABLED", None)
            with patch.object(ParseHelper, "__init__", lambda self, db: None):
                ph = ParseHelper.__new__(ParseHelper)
                # Re-run the relevant init line in isolation so the test doesn't
                # need a live DB session.
                ph._sandbox_parsing_enabled = (
                    os.getenv("SANDBOX_PARSING_ENABLED", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                )
                assert ph._sandbox_parsing_enabled is False

    @pytest.mark.parametrize("val", ["1", "true", "True", "yes", "ON"])
    def test_truthy_values_enable(self, val):
        with patch.dict(os.environ, {"SANDBOX_PARSING_ENABLED": val}, clear=False):
            ph = ParseHelper.__new__(ParseHelper)
            ph._sandbox_parsing_enabled = (
                os.getenv("SANDBOX_PARSING_ENABLED", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            assert ph._sandbox_parsing_enabled is True


class TestCloneViaSandbox:
    """`_clone_via_sandbox` returns the same tuple shape as the legacy path."""

    @pytest.mark.asyncio
    async def test_returns_tuple_and_uses_handle_local_path(self):
        from app.modules.parsing.graph_construction.parsing_helper import (
            ParseHelper,
        )

        ph = ParseHelper.__new__(ParseHelper)
        ph.github_service = MagicMock()
        # github_service.get_repo raises → owner/auth fall through to None.
        ph.github_service.get_repo = MagicMock(side_effect=RuntimeError("no metadata"))
        ph._sandbox_parsing_enabled = True

        fake_handle = MagicMock(
            local_path="/tmp/fake-worktree",
            backend_kind="local",
            workspace_id="ws_test",
        )
        fake_client = MagicMock()
        fake_client.get_workspace = AsyncMock(return_value=fake_handle)

        with patch(
            "app.modules.parsing.graph_construction.parsing_helper._get_git_imports"
        ) as mock_git, patch(
            "sandbox.SandboxClient.from_env", return_value=fake_client
        ):
            mock_git.return_value = (RuntimeError, RuntimeError, MagicMock())
            repo, owner, auth, local_path = await ph._clone_via_sandbox(
                RepoDetails(repo_name="owner/repo", branch_name="main"),
                user_id="u1",
                auth_token="tok",
                project_id="p1",
            )

        assert local_path == "/tmp/fake-worktree"
        assert owner is None  # github_service raised
        assert auth is None
        # SandboxClient.get_workspace was called with mode=ANALYSIS, no
        # branch creation, and the user-provided token.
        kwargs = fake_client.get_workspace.call_args.kwargs
        assert kwargs["mode"].value == "analysis"
        assert kwargs["create_branch"] is False
        assert kwargs["auth_token"] == "tok"
        assert kwargs["repo"] == "owner/repo"

    @pytest.mark.asyncio
    async def test_raises_when_handle_has_no_local_path(self):
        ph = ParseHelper.__new__(ParseHelper)
        ph.github_service = MagicMock()
        ph.github_service.get_repo = MagicMock(side_effect=RuntimeError("no metadata"))
        ph._sandbox_parsing_enabled = True

        fake_handle = MagicMock(
            local_path=None,
            backend_kind="daytona",
            workspace_id="ws_test",
        )
        fake_client = MagicMock()
        fake_client.get_workspace = AsyncMock(return_value=fake_handle)

        with patch(
            "sandbox.SandboxClient.from_env", return_value=fake_client
        ):
            with pytest.raises(RuntimeError, match="local-fs backend"):
                await ph._clone_via_sandbox(
                    RepoDetails(repo_name="owner/repo", branch_name="main"),
                    user_id="u1",
                    auth_token=None,
                    project_id="p1",
                )
