"""Rich episode body formatters for Graphiti ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.episode_formatters import build_commit_episode, build_pr_episode

pytestmark = pytest.mark.unit


# --- build_pr_episode ------------------------------------------------------


class TestBuildPrEpisode:
    def test_minimal_pr_renders_with_defaults(self) -> None:
        ep = build_pr_episode(
            pr_data={"number": 42, "title": "Fix flake"},
            commits=[],
            review_threads=[],
            linked_issues=[],
        )
        assert ep["name"] == "pr_42_merged"
        assert ep["source_id"] == "pr_42_merged"
        assert ep["source_description"] == "GitHub Pull Request #42"
        assert "PR #42: Fix flake" in ep["episode_body"]
        # Default sections appear with "- None" placeholders.
        assert "Files changed: None" in ep["episode_body"]
        assert "RELATED ISSUES:\n- None" in ep["episode_body"]
        assert "COMMITS:\n- None" in ep["episode_body"]
        assert "REVIEW DISCUSSIONS:\n- None" in ep["episode_body"]
        assert "PR ISSUE COMMENTS:\n- None" in ep["episode_body"]
        assert "Labels: None" in ep["episode_body"]
        assert "Milestone/Feature: None" in ep["episode_body"]
        assert "No PR description provided." in ep["episode_body"]

    def test_full_pr_renders_all_sections(self) -> None:
        ep = build_pr_episode(
            pr_data={
                "number": 7,
                "title": "Add auth",
                "author": "alice",
                "head_branch": "feat/auth",
                "base_branch": "main",
                "merged_at": "2026-01-02T03:04:05Z",
                "body": "Adds OAuth flow.",
                "files": [{"filename": "app/auth.py"}, {"filename": ""}],
                "labels": ["enhancement", {"name": "security"}, None],
                "milestone": {"title": "Q1 release"},
            },
            commits=[
                {"sha": "abc123", "message": "init\nbody"},
                {"sha": "def456", "message": "  finish  "},
            ],
            review_threads=[
                {
                    "path": "app/auth.py",
                    "line": 10,
                    "diff_hunk": "@@",
                    "comments": [
                        {"author": "bob", "body": "looks good"},
                    ],
                }
            ],
            linked_issues=[
                {"number": 12, "title": "Need auth", "body": "context goes here"},
            ],
            issue_comments=[
                {"user": {"login": "carol"}, "body": "+1"},
                {"user": "not-a-dict", "body": "weird"},  # falls through gracefully
            ],
        )
        body = ep["episode_body"]
        assert "Author: alice" in body
        assert "From branch: feat/auth" in body
        assert "To branch: main" in body
        assert "Merged at: 2026-01-02T03:04:05Z" in body
        assert "Files changed: app/auth.py" in body
        assert "- abc123: init" in body  # only first line of multi-line message
        assert "- def456: finish" in body
        assert "- File: app/auth.py, Line: 10" in body
        assert "  - bob: looks good" in body
        assert "- #12: Need auth | context goes here" in body
        assert "- carol: +1" in body
        # Mixed labels (string, dict, None) are joined cleanly.
        assert "Labels: enhancement, security" in body
        assert "Milestone/Feature: Q1 release" in body

    def test_milestone_string_is_rendered_as_is(self) -> None:
        ep = build_pr_episode(
            pr_data={"number": 1, "milestone": "Phase 8"},
            commits=[],
            review_threads=[],
            linked_issues=[],
        )
        assert "Milestone/Feature: Phase 8" in ep["episode_body"]

    def test_reference_time_falls_back_to_updated_at(self) -> None:
        # No merged_at; updated_at parses cleanly.
        ep = build_pr_episode(
            pr_data={"number": 1, "updated_at": "2026-04-27T00:00:00+00:00"},
            commits=[],
            review_threads=[],
            linked_issues=[],
        )
        assert isinstance(ep["reference_time"], datetime)
        assert ep["reference_time"].year == 2026

    def test_reference_time_handles_datetime_value(self) -> None:
        dt = datetime(2026, 4, 27, tzinfo=timezone.utc)
        ep = build_pr_episode(
            pr_data={"number": 1, "merged_at": dt},
            commits=[],
            review_threads=[],
            linked_issues=[],
        )
        assert ep["reference_time"] == dt

    def test_invalid_iso_falls_back_to_now(self) -> None:
        ep = build_pr_episode(
            pr_data={"number": 1, "merged_at": "not-a-date"},
            commits=[],
            review_threads=[],
            linked_issues=[],
        )
        # Falls back to ``datetime.utcnow()`` — just check it's a datetime.
        assert isinstance(ep["reference_time"], datetime)

    def test_long_issue_body_truncated_to_280_chars(self) -> None:
        ep = build_pr_episode(
            pr_data={"number": 1},
            commits=[],
            review_threads=[],
            linked_issues=[
                {"number": 5, "title": "T", "body": "x" * 500},
            ],
        )
        # The body lands inline on the issue line; only the first 280 chars survive.
        line = next(l for l in ep["episode_body"].splitlines() if l.startswith("- #5:"))
        # title (1 char) + " | " + 280 x's
        assert line.count("x") == 280


# --- build_commit_episode --------------------------------------------------


class TestBuildCommitEpisode:
    def test_full_commit_episode(self) -> None:
        ep = build_commit_episode(
            {
                "sha": "abc123",
                "message": "  fix bug  ",
                "author": "alice",
                "committed_at": "2026-04-27T00:00:00+00:00",
            },
            branch="main",
        )
        assert ep["name"] == "commit_abc123"
        assert ep["source_id"] == "commit_abc123"
        assert ep["source_description"] == "GitHub Commit abc123"
        body = ep["episode_body"]
        assert "Standalone commit abc123" in body
        assert "Author: alice" in body
        assert "Branch: main" in body
        assert "Committed at: 2026-04-27T00:00:00+00:00" in body
        assert "Message:\nfix bug" in body
        assert isinstance(ep["reference_time"], datetime)

    def test_missing_sha_uses_unknown(self) -> None:
        ep = build_commit_episode({}, branch="main")
        assert ep["name"] == "commit_unknown"
        assert "Standalone commit unknown" in ep["episode_body"]

    def test_datetime_committed_at_renders_iso(self) -> None:
        dt = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        ep = build_commit_episode({"sha": "x", "committed_at": dt}, branch="main")
        assert "Committed at: 2026-04-27T12:00:00+00:00" in ep["episode_body"]

    def test_missing_committed_at_renders_empty(self) -> None:
        ep = build_commit_episode({"sha": "x"}, branch="main")
        assert "Committed at: " in ep["episode_body"]
