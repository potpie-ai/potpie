"""Unit tests for context-graph deterministic parsers."""

import pytest

from domain.deterministic_extractors import (
    extract_feature_from_labels,
    extract_issue_refs,
    extract_ticket_from_branch,
    parse_diff_hunks,
)
from domain.review_thread_grouper import group_review_threads

pytestmark = pytest.mark.unit


class TestDeterministicExtractors:
    def test_extract_issue_refs(self):
        text = "Fixes #12. Also closes #34 and resolves #12 again."
        assert extract_issue_refs(text) == [12, 34]

    def test_extract_ticket_from_branch(self):
        assert extract_ticket_from_branch("feature/PLAT-123-add-intelligence") == "PLAT-123"
        assert extract_ticket_from_branch("chore/no-ticket") is None

    def test_extract_feature_from_milestone_first(self):
        labels = [{"name": "bug"}, {"name": "context-graph"}]
        milestone = {"title": "Context Intelligence"}
        assert extract_feature_from_labels(labels, milestone) == "Context Intelligence"

    def test_extract_feature_from_labels_fallback(self):
        labels = ["bug", "docs", "payments"]
        assert extract_feature_from_labels(labels, None) == "payments"

    def test_parse_diff_hunks(self):
        patch = (
            "@@ -10,2 +20,5 @@\n"
            " old\n"
            " new\n"
            "@@ -40 +70 @@\n"
            " line\n"
            "@@ -100,1 +200,0 @@\n"
        )
        assert parse_diff_hunks(patch) == [(20, 24), (70, 70)]


class TestReviewThreadGrouper:
    def test_group_review_threads(self):
        comments = [
            {
                "id": 1,
                "body": "Root comment",
                "path": "app/main.py",
                "line": 42,
                "diff_hunk": "@@ -1 +1 @@",
                "user": {"login": "alice"},
                "created_at": "2026-03-24T10:00:00Z",
                "in_reply_to_id": None,
            },
            {
                "id": 2,
                "body": "Reply",
                "path": "app/main.py",
                "line": 42,
                "diff_hunk": "@@ -1 +1 @@",
                "user": {"login": "bob"},
                "created_at": "2026-03-24T10:01:00Z",
                "in_reply_to_id": 1,
            },
            {
                "id": 3,
                "body": "Separate thread",
                "path": "app/core/config_provider.py",
                "line": 10,
                "diff_hunk": "@@ -5 +5 @@",
                "user": {"login": "carol"},
                "created_at": "2026-03-24T10:02:00Z",
                "in_reply_to_id": None,
            },
        ]

        threads = group_review_threads(comments)
        assert len(threads) == 2

        first = threads[0]
        assert first["thread_id"] == 1
        assert first["path"] == "app/main.py"
        assert first["line"] == 42
        assert [c["author"] for c in first["comments"]] == ["alice", "bob"]

        second = threads[1]
        assert second["thread_id"] == 3
        assert second["path"] == "app/core/config_provider.py"
