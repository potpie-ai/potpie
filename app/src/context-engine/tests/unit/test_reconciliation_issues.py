"""Split validation lines into {entity, issue} rows."""

from domain.reconciliation_issues import validation_line_to_issue, validation_lines_to_issues


def test_unknown_canonical_labels_entity_key_with_colons() -> None:
    line = "adr:0042: unknown canonical labels: ADR"
    assert validation_line_to_issue(line) == {
        "entity": "adr:0042",
        "issue": "unknown canonical labels: ADR",
    }


def test_missing_required_properties_with_type_label() -> None:
    line = "adr:0042:Decision: missing required properties: summary"
    assert validation_line_to_issue(line) == {
        "entity": "adr:0042:Decision",
        "issue": "missing required properties: summary",
    }


def test_invalid_lifecycle() -> None:
    line = (
        "adr:0042:Decision: invalid lifecycle/status 'recorded'; "
        "allowed: accepted, proposed, rejected, superseded, unknown"
    )
    out = validation_line_to_issue(line)
    assert out["entity"] == "adr:0042:Decision"
    assert "invalid lifecycle/status" in out["issue"]
    assert "recorded" in out["issue"]


def test_unknown_edge_type() -> None:
    line = "DECIDED_BY: unknown canonical edge type"
    assert validation_line_to_issue(line) == {
        "entity": "DECIDED_BY",
        "issue": "unknown canonical edge type",
    }


def test_validation_lines_to_issues_order() -> None:
    lines = [
        "adr:0042: unknown canonical labels: ADR",
        "technology:mongodb: unknown canonical labels: Database, Technology",
    ]
    rows = validation_lines_to_issues(lines)
    assert len(rows) == 2
    assert rows[0]["entity"] == "adr:0042"
    assert rows[1]["entity"] == "technology:mongodb"

