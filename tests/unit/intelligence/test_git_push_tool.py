"""Unit tests for GitPushTool.

Covers the credential-hygiene fix for issue #725: after a push that
injected a token-bearing remote URL, restoring the plain URL must NOT
silently swallow exceptions, otherwise the token can persist in
``.git/config`` indefinitely.
"""
from unittest.mock import MagicMock

import pytest
from git.exc import GitCommandError
from loguru import logger as _loguru_logger

from app.modules.intelligence.tools.code_query_tools.git_push_tool import (
    GitPushTool,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def tool():
    sql_db = MagicMock()
    # Bypass __init__ side-effects (RepoManager, ProjectService, env reads).
    instance = GitPushTool.__new__(GitPushTool)
    instance.sql_db = sql_db
    instance.user_id = "u-test"
    instance.project_service = MagicMock()
    instance.repo_manager = MagicMock()
    return instance


@pytest.fixture
def loguru_records():
    """Capture loguru records — caplog is stdlib-only and won't see them."""
    records = []
    sink_id = _loguru_logger.add(
        lambda msg: records.append(msg.record), level="DEBUG"
    )
    try:
        yield records
    finally:
        _loguru_logger.remove(sink_id)


def _critical_messages(records):
    return [r["message"] for r in records if r["level"].name == "CRITICAL"]


def _git_command_error(msg: str) -> GitCommandError:
    return GitCommandError(["git", "remote", "set-url"], 128, stderr=msg)


def test_restore_plain_remote_url_happy_path_does_not_log_critical(
    tool, loguru_records
):
    repo = MagicMock()

    tool._restore_plain_remote_url(
        repo, "origin", "https://github.com/x/y.git"
    )

    repo.git.remote.assert_called_once_with(
        "set-url", "origin", "https://github.com/x/y.git"
    )
    repo.git.config.assert_not_called()
    assert _critical_messages(loguru_records) == []


def test_restore_plain_remote_url_falls_back_to_git_config_on_set_url_failure(
    tool, loguru_records
):
    """If `git remote set-url` fails we MUST surface a critical log AND try
    to overwrite the URL directly via `git config`, so the token-bearing
    URL does not silently persist in .git/config (issue #725)."""
    repo = MagicMock()
    repo.git.remote.side_effect = _git_command_error("locked config")
    repo.git.config.return_value = ""

    tool._restore_plain_remote_url(
        repo, "origin", "https://github.com/x/y.git"
    )

    repo.git.remote.assert_called_once_with(
        "set-url", "origin", "https://github.com/x/y.git"
    )
    repo.git.config.assert_called_once_with(
        "remote.origin.url", "https://github.com/x/y.git"
    )

    crit = _critical_messages(loguru_records)
    assert crit, "expected at least one CRITICAL log on restore failure"
    assert any("SECURITY" in m for m in crit)
    assert any("origin" in m for m in crit)


def test_restore_plain_remote_url_logs_double_critical_when_fallback_also_fails(
    tool, loguru_records
):
    """Both restore attempts fail — operator must see TWO CRITICAL log
    entries so escalation is unambiguous."""
    repo = MagicMock()
    repo.git.remote.side_effect = _git_command_error("locked config")
    repo.git.config.side_effect = _git_command_error("also locked")

    tool._restore_plain_remote_url(
        repo, "origin", "https://github.com/x/y.git"
    )

    crit = _critical_messages(loguru_records)
    assert len(crit) >= 2, (
        f"expected >=2 CRITICAL logs on double failure, got: {crit!r}"
    )
    assert any("manually remove" in m for m in crit), (
        f"expected operator-action hint, got: {crit!r}"
    )


def test_restore_plain_remote_url_does_not_swallow_silently(
    tool, loguru_records
):
    """Pre-fix regression guard: under no failure mode does the helper
    return without any log output."""
    repo = MagicMock()
    repo.git.remote.side_effect = _git_command_error("fail")
    repo.git.config.side_effect = _git_command_error("fail2")

    tool._restore_plain_remote_url(repo, "origin", "https://example/x.git")

    assert loguru_records, "helper must not silently swallow failures"
