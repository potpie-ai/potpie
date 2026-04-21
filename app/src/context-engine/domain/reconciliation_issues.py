"""Split ontology / plan validation lines into {entity, issue} for API + CLI surfaces."""

from __future__ import annotations


def validation_lines_to_issues(lines: list[str]) -> list[dict[str, str]]:
    """Map validation error strings to structured rows for HTTP/CLI."""
    return [validation_line_to_issue(line) for line in lines]


def validation_line_to_issue(line: str) -> dict[str, str]:
    """Split one error line into entity ref and human-readable issue text.

    Formats are produced by ``domain.ontology`` and ``reconciliation_validation``.
    """
    s = line.strip()
    if not s:
        return {"entity": "", "issue": s}

    # Order: longest / most specific markers first so colons in entity keys match correctly.
    pairs: list[tuple[str, str]] = [
        (": missing required properties:", "missing required properties:"),
        (": invalid lifecycle/status ", "invalid lifecycle/status "),
        (": unknown canonical labels:", "unknown canonical labels:"),
        (": at least one public canonical label is required", "at least one public canonical label is required"),
        (": at least one label is required", "at least one label is required"),
        (": unknown canonical edge type", "unknown canonical edge type"),
        (": from_entity_key is required", "from_entity_key is required"),
        (": to_entity_key is required", "to_entity_key is required"),
        (": invalid endpoint labels ", "invalid endpoint labels "),
    ]
    for needle, issue_prefix in pairs:
        if needle in s:
            ent, _, rest = s.partition(needle)
            return {"entity": ent.strip(), "issue": (issue_prefix + rest).strip()}

    # Invalidation / misc messages without ``entity:`` prefix
    if s.startswith("invalidation "):
        return {"entity": "", "issue": s}

    if s == "entity_key is required":
        return {"entity": "", "issue": s}

    return {"entity": "", "issue": s}
