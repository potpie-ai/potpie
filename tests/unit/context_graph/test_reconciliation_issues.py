"""Validation error string → {entity, issue} structured row parsing."""

from __future__ import annotations

import pytest

from domain.reconciliation_issues import (
    validation_line_to_issue,
    validation_lines_to_issues,
)

pytestmark = pytest.mark.unit


class TestValidationLineToIssue:
    def test_empty_line_returns_empty_entity(self) -> None:
        assert validation_line_to_issue("") == {"entity": "", "issue": ""}

    def test_whitespace_only_returns_empty_entity(self) -> None:
        # strip() collapses to empty, treated as the empty case.
        assert validation_line_to_issue("   \n").get("entity") == ""

    def test_missing_required_properties(self) -> None:
        line = "github:pr:123: missing required properties: title, body"
        out = validation_line_to_issue(line)
        assert out["entity"] == "github:pr:123"
        assert out["issue"] == "missing required properties: title, body"

    def test_invalid_lifecycle_status(self) -> None:
        out = validation_line_to_issue("github:issue:42: invalid lifecycle/status weird")
        assert out["entity"] == "github:issue:42"
        assert out["issue"].startswith("invalid lifecycle/status")
        assert "weird" in out["issue"]

    def test_unknown_canonical_labels(self) -> None:
        out = validation_line_to_issue("ent: unknown canonical labels: FooBar")
        assert out["entity"] == "ent"
        assert out["issue"] == "unknown canonical labels: FooBar"

    def test_at_least_one_public_canonical_label(self) -> None:
        out = validation_line_to_issue(
            "ent_x: at least one public canonical label is required"
        )
        assert out["entity"] == "ent_x"
        assert out["issue"] == "at least one public canonical label is required"

    def test_at_least_one_label_required(self) -> None:
        out = validation_line_to_issue("ent_y: at least one label is required")
        assert out["entity"] == "ent_y"
        assert out["issue"] == "at least one label is required"

    def test_unknown_canonical_edge_type(self) -> None:
        out = validation_line_to_issue("a→b: unknown canonical edge type WEIRD")
        assert out["entity"] == "a→b"
        assert out["issue"].startswith("unknown canonical edge type")

    def test_from_entity_key_required(self) -> None:
        out = validation_line_to_issue("edge:42: from_entity_key is required")
        assert out["entity"] == "edge:42"
        assert out["issue"] == "from_entity_key is required"

    def test_to_entity_key_required(self) -> None:
        out = validation_line_to_issue("edge:42: to_entity_key is required")
        assert out["entity"] == "edge:42"
        assert out["issue"] == "to_entity_key is required"

    def test_invalid_endpoint_labels(self) -> None:
        out = validation_line_to_issue("RELATES_TO: invalid endpoint labels (Foo, Bar)")
        assert out["entity"] == "RELATES_TO"
        assert out["issue"].startswith("invalid endpoint labels")

    def test_invalidation_message_has_no_entity(self) -> None:
        out = validation_line_to_issue("invalidation target missing")
        assert out == {"entity": "", "issue": "invalidation target missing"}

    def test_entity_key_required_message(self) -> None:
        out = validation_line_to_issue("entity_key is required")
        assert out == {"entity": "", "issue": "entity_key is required"}

    def test_uncategorized_message_falls_through(self) -> None:
        out = validation_line_to_issue("some entirely unknown error string")
        assert out == {"entity": "", "issue": "some entirely unknown error string"}

    def test_entity_key_with_colon_is_preserved(self) -> None:
        # Entity keys often contain colons (github:pr:123). The matcher uses the
        # marker phrase, not just the first colon, so the entity is preserved.
        out = validation_line_to_issue(
            "github:pr:99: missing required properties: title"
        )
        assert out["entity"] == "github:pr:99"


class TestValidationLinesToIssues:
    def test_empty_input(self) -> None:
        assert validation_lines_to_issues([]) == []

    def test_maps_each_line_independently(self) -> None:
        result = validation_lines_to_issues(
            [
                "ent_a: missing required properties: title",
                "entity_key is required",
            ]
        )
        assert len(result) == 2
        assert result[0]["entity"] == "ent_a"
        assert result[1] == {"entity": "", "issue": "entity_key is required"}
