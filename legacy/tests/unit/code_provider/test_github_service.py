"""Unit tests for GithubService (executor and token initialization)."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from app.modules.code_provider.github.github_service import GithubService


pytestmark = pytest.mark.unit


class TestGithubServiceExecutor:
    def test_get_executor_returns_executor(self):
        try:
            executor = GithubService._get_executor()
            assert executor is not None
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            GithubService.shutdown_executor()

    def test_get_executor_cached(self):
        try:
            e1 = GithubService._get_executor()
            e2 = GithubService._get_executor()
            assert e1 is e2
        finally:
            GithubService.shutdown_executor()

    def test_shutdown_executor_clears_shared(self):
        GithubService._get_executor()
        GithubService.shutdown_executor()
        assert GithubService._shared_executor is None
        # Next _get_executor creates a new one
        e = GithubService._get_executor()
        assert e is not None
        GithubService.shutdown_executor()


class TestGithubServiceInitializeTokens:
    def test_initialize_tokens_empty_raises(self):
        with patch.dict("os.environ", {"GH_TOKEN_LIST": ""}, clear=False):
            GithubService.gh_token_list = []
            with pytest.raises(ValueError, match="empty or not set"):
                GithubService.initialize_tokens()

    def test_initialize_tokens_sets_list(self):
        with patch.dict("os.environ", {"GH_TOKEN_LIST": "token1,token2 , token3"}, clear=False):
            GithubService.gh_token_list = []
            GithubService.initialize_tokens()
            assert len(GithubService.gh_token_list) == 3
            assert GithubService.gh_token_list[0] == "token1"
            assert GithubService.gh_token_list[1] == "token2"
            assert GithubService.gh_token_list[2] == "token3"
            # Restore for other tests that may expect tokens
            GithubService.gh_token_list = []
