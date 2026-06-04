"""Deterministic regex extractors for context-graph ingestion."""

from __future__ import annotations

import pytest

from domain.deterministic_extractors import (
    extract_feature_from_labels,
    extract_issue_refs,
    extract_ticket_from_branch,
    parse_diff_hunks,
)

pytestmark = pytest.mark.unit


class TestExtractIssueRefs:
    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("Fixes #42", [42]),
            ("fix #1", [1]),
            ("Fixed #5", [5]),
            ("Closes #99", [99]),
            ("close #7", [7]),
            ("closed #2", [2]),
            ("Resolves #128", [128]),
            ("resolve #3", [3]),
            ("resolved #4", [4]),
            ("Fixes: #50", [50]),
        ],
    )
    def test_supported_verbs(self, phrase: str, expected: list[int]) -> None:
        assert extract_issue_refs(phrase) == expected

    def test_extracts_multiple_refs_sorted_unique(self) -> None:
        text = "Fixes #5 and closes #2 and also resolves #5"
        assert extract_issue_refs(text) == [2, 5]

    def test_no_verb_no_match(self) -> None:
        assert extract_issue_refs("see #42 for context") == []

    def test_none_returns_empty(self) -> None:
        assert extract_issue_refs(None) == []

    def test_empty_returns_empty(self) -> None:
        assert extract_issue_refs("") == []

    def test_case_insensitive(self) -> None:
        assert extract_issue_refs("FIXES #11") == [11]


class TestExtractTicketFromBranch:
    @pytest.mark.parametrize(
        "branch,expected",
        [
            ("feature/PROJ-123-add-thing", "PROJ-123"),
            ("PROJ-1", "PROJ-1"),
            ("AB12-9", "AB12-9"),  # uppercase letters then digits in prefix
        ],
    )
    def test_extracts_first_ticket(self, branch: str, expected: str) -> None:
        assert extract_ticket_from_branch(branch) == expected

    def test_returns_first_match_when_multiple(self) -> None:
        assert extract_ticket_from_branch("feature/PROJ-1-and-PROJ-2") == "PROJ-1"

    def test_no_ticket(self) -> None:
        assert extract_ticket_from_branch("main") is None
        assert extract_ticket_from_branch("feature/lowercase-only") is None

    def test_none_returns_none(self) -> None:
        assert extract_ticket_from_branch(None) is None

    def test_empty_returns_none(self) -> None:
        assert extract_ticket_from_branch("") is None


class TestExtractFeatureFromLabels:
    def test_milestone_string_takes_priority(self) -> None:
        assert extract_feature_from_labels(["bug"], milestone="Q3 Roadmap") == "Q3 Roadmap"

    def test_milestone_dict_with_title(self) -> None:
        ms = {"title": "Phase 8"}
        assert extract_feature_from_labels(["bug"], milestone=ms) == "Phase 8"

    def test_milestone_dict_without_title_falls_through_to_labels(self) -> None:
        ms = {"title": ""}
        assert extract_feature_from_labels(["my-feature"], milestone=ms) == "my-feature"

    def test_whitespace_milestone_falls_through_to_labels(self) -> None:
        # Whitespace-only string is rejected by ``strip()`` check.
        assert extract_feature_from_labels(["x"], milestone="   ") == "x"

    def test_excluded_labels_skipped(self) -> None:
        assert extract_feature_from_labels(["bug", "chore", "feature-x"]) == "feature-x"

    @pytest.mark.parametrize(
        "label",
        ["bug", "type: bug", "fix", "hotfix", "chore", "docs", "documentation",
         "refactor", "test"],
    )
    def test_each_excluded_label_skipped(self, label: str) -> None:
        assert extract_feature_from_labels([label]) is None

    def test_dict_labels_pull_name(self) -> None:
        labels = [{"name": "bug"}, {"name": "auth"}]
        assert extract_feature_from_labels(labels) == "auth"

    def test_unknown_label_types_skipped(self) -> None:
        # Numbers, sets, etc. are not strings/dicts → skipped.
        assert extract_feature_from_labels([42, {"name": "real"}]) == "real"

    def test_none_milestone_and_labels(self) -> None:
        assert extract_feature_from_labels(None) is None

    def test_empty_labels(self) -> None:
        assert extract_feature_from_labels([]) is None

    def test_case_insensitive_exclusion(self) -> None:
        # Excluded comparison is lower()-cased.
        assert extract_feature_from_labels(["BUG", "good-feature"]) == "good-feature"


class TestParseDiffHunks:
    def test_parses_single_hunk(self) -> None:
        patch = "@@ -1,3 +5,4 @@\n some\n+context\n more\n"
        assert parse_diff_hunks(patch) == [(5, 8)]

    def test_parses_default_count_when_missing(self) -> None:
        # ``+10`` without ``,N`` means count=1 → range (10, 10).
        assert parse_diff_hunks("@@ -0,0 +10 @@\n+x") == [(10, 10)]

    def test_parses_multiple_hunks(self) -> None:
        patch = (
            "@@ -1,2 +1,2 @@\n"
            "some\n"
            "@@ -10,3 +20,5 @@\n"
            "more\n"
        )
        assert parse_diff_hunks(patch) == [(1, 2), (20, 24)]

    def test_zero_count_skipped(self) -> None:
        assert parse_diff_hunks("@@ -1,1 +0,0 @@\n removed") == []

    def test_none_returns_empty(self) -> None:
        assert parse_diff_hunks(None) == []

    def test_empty_returns_empty(self) -> None:
        assert parse_diff_hunks("") == []

    def test_no_hunk_header_returns_empty(self) -> None:
        assert parse_diff_hunks("just some plain text") == []
