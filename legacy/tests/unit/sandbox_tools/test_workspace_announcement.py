"""Tests for the per-run workspace setup announcement.

The first sandbox tool call in a fresh run should surface a one-line
"Setting up workspace for ... " banner. Subsequent calls in the same run
must be silent. Resetting the run context (i.e. starting a new agent run)
must re-arm the banner.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from app.modules.intelligence.tools.sandbox import client as sandbox_client_mod
from app.modules.intelligence.tools.sandbox.client import (
    clear_project_summary_cache,
    lookup_project_summary,
    workspace_setup_banner,
)
from app.modules.intelligence.tools.sandbox.context import (
    is_workspace_announced,
    mark_workspace_announced,
    set_run_context,
)


_PROJECT_SUMMARY = {
    "repo_name": "owner/repo",
    "base_branch": "main",
    "repo_url": "https://github.com/owner/repo.git",
    "user_id": "u1",
}


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Each test starts with a clean cache + run-context state."""
    clear_project_summary_cache()
    set_run_context(user_id="u1", conversation_id="conv-12345678abcd", branch="main")
    yield
    clear_project_summary_cache()


# ----------------------------------------------------------------------
# lookup_project_summary
# ----------------------------------------------------------------------
def test_lookup_project_summary_caches_db_read() -> None:
    calls: list[str] = []

    def _fake_get(_pid: str) -> dict[str, Any]:
        calls.append(_pid)
        return {
            "project_name": "owner/repo",
            "branch_name": "main",
            "repo_path": "https://github.com/owner/repo.git",
            "user_id": "u1",
        }

    class _StubService:
        def __init__(self, db: Any) -> None:
            self._db = db

        def get_project_from_db_by_id_sync(self, pid: str) -> dict[str, Any]:
            return _fake_get(pid)

    class _StubSession:
        def __enter__(self) -> "_StubSession":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    with patch(
        "app.modules.projects.projects_service.ProjectService", _StubService
    ), patch("app.core.database.SessionLocal", _StubSession):
        first = lookup_project_summary("p1")
        second = lookup_project_summary("p1")

    assert first == _PROJECT_SUMMARY
    assert second is first  # same dict instance — cache hit
    assert calls == ["p1"]  # DB was hit exactly once


def test_lookup_project_summary_raises_for_missing() -> None:
    class _StubService:
        def __init__(self, db: Any) -> None:
            self._db = db

        def get_project_from_db_by_id_sync(self, pid: str) -> dict[str, Any] | None:
            return None

    class _StubSession:
        def __enter__(self) -> "_StubSession":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    with patch(
        "app.modules.projects.projects_service.ProjectService", _StubService
    ), patch("app.core.database.SessionLocal", _StubSession):
        with pytest.raises(ValueError, match="Cannot find repo details"):
            lookup_project_summary("does-not-exist")


# ----------------------------------------------------------------------
# workspace_setup_banner
# ----------------------------------------------------------------------
def test_banner_includes_repo_and_derived_branch() -> None:
    with patch.object(
        sandbox_client_mod, "lookup_project_summary", return_value=_PROJECT_SUMMARY
    ):
        banner = workspace_setup_banner("p1", conversation_id="conv-12345678abcd")
    # The derived branch is agent/edits-{first 8 chars of conversation_id}.
    assert banner == (
        "Setting up workspace for owner/repo on agent/edits-conv-123 (from main)…"
    )


def test_banner_without_conversation_id_falls_back() -> None:
    with patch.object(
        sandbox_client_mod, "lookup_project_summary", return_value=_PROJECT_SUMMARY
    ):
        banner = workspace_setup_banner("p1", conversation_id=None)
    assert banner == "Setting up workspace for owner/repo (from main)…"


def test_banner_returns_empty_when_lookup_fails() -> None:
    with patch.object(
        sandbox_client_mod,
        "lookup_project_summary",
        side_effect=ValueError("boom"),
    ):
        banner = workspace_setup_banner("missing", conversation_id="c1")
    assert banner == ""


# ----------------------------------------------------------------------
# announced contextvar lifecycle
# ----------------------------------------------------------------------
def test_set_run_context_resets_announced_flag() -> None:
    mark_workspace_announced()
    assert is_workspace_announced() is True
    set_run_context(user_id="u1", conversation_id="conv-new")
    assert is_workspace_announced() is False


def test_mark_announced_is_idempotent() -> None:
    mark_workspace_announced()
    mark_workspace_announced()
    assert is_workspace_announced() is True


# ----------------------------------------------------------------------
# Streaming hook: _sandbox_setup_banner_if_needed
# ----------------------------------------------------------------------
def test_streaming_hook_announces_only_once_per_run() -> None:
    from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
        _sandbox_setup_banner_if_needed,
    )

    with patch.object(
        sandbox_client_mod, "lookup_project_summary", return_value=_PROJECT_SUMMARY
    ):
        first = _sandbox_setup_banner_if_needed(
            "sandbox_text_editor", {"project_id": "p1"}
        )
        second = _sandbox_setup_banner_if_needed(
            "sandbox_shell", {"project_id": "p1"}
        )
    assert first.startswith("Setting up workspace for owner/repo")
    assert second == ""  # subsequent sandbox calls in the same run are silent


def test_streaming_hook_skips_non_sandbox_tools() -> None:
    from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
        _sandbox_setup_banner_if_needed,
    )

    out = _sandbox_setup_banner_if_needed("get_code_from_node_id", {"project_id": "p1"})
    assert out == ""
    assert is_workspace_announced() is False


def test_streaming_hook_skips_when_project_id_absent() -> None:
    from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
        _sandbox_setup_banner_if_needed,
    )

    out = _sandbox_setup_banner_if_needed("sandbox_search", {})
    assert out == ""
    assert is_workspace_announced() is False


def test_new_run_re_arms_announcement() -> None:
    """Per-conversation isolation: a fresh ``set_run_context`` call must
    re-enable the banner so the next conversation announces its own
    workspace."""
    from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
        _sandbox_setup_banner_if_needed,
    )

    with patch.object(
        sandbox_client_mod, "lookup_project_summary", return_value=_PROJECT_SUMMARY
    ):
        first_run = _sandbox_setup_banner_if_needed(
            "sandbox_text_editor", {"project_id": "p1"}
        )
        # Within the same run, no further announcement.
        within = _sandbox_setup_banner_if_needed(
            "sandbox_git", {"project_id": "p1"}
        )
        # Simulate the start of a new agent run.
        set_run_context(user_id="u1", conversation_id="conv-different")
        next_run = _sandbox_setup_banner_if_needed(
            "sandbox_shell", {"project_id": "p1"}
        )
    assert first_run.startswith("Setting up workspace")
    assert within == ""
    assert next_run.startswith("Setting up workspace")
