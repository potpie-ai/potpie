"""Deterministic parsing helpers for context graph ingestion."""

import re
from typing import Any, Optional

ISSUE_REF_PATTERN = re.compile(
    r"(?i)\b(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?)\b\s*:?\s*#(\d+)\b"
)
TICKET_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
HUNK_PATTERN = re.compile(r"^@@\s*-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s*@@", re.MULTILINE)


def extract_issue_refs(text: Optional[str]) -> list[int]:
    """Extract issue refs from 'Fixes #123' style clauses."""
    if not text:
        return []
    refs = [int(match.group(1)) for match in ISSUE_REF_PATTERN.finditer(text)]
    return sorted(set(refs))


def extract_ticket_from_branch(branch_name: Optional[str]) -> Optional[str]:
    """Extract ticket keys like PROJ-123 from branch names."""
    if not branch_name:
        return None
    match = TICKET_PATTERN.search(branch_name)
    return match.group(1) if match else None


def extract_feature_from_labels(
    labels: Optional[list[Any]],
    milestone: Optional[Any] = None,
) -> Optional[str]:
    """
    Determine feature area from milestone first, then labels.
    Ignores typical bug/chore labels when selecting a feature-like label.
    """
    if isinstance(milestone, str) and milestone.strip():
        return milestone.strip()
    if isinstance(milestone, dict):
        title = (milestone.get("title") or "").strip()
        if title:
            return title

    if not labels:
        return None

    excluded = {
        "bug",
        "type: bug",
        "fix",
        "hotfix",
        "chore",
        "docs",
        "documentation",
        "refactor",
        "test",
    }
    for item in labels:
        if isinstance(item, str):
            label_name = item.strip()
        elif isinstance(item, dict):
            label_name = (item.get("name") or "").strip()
        else:
            continue
        if label_name and label_name.lower() not in excluded:
            return label_name
    return None


def parse_diff_hunks(patch: Optional[str]) -> list[tuple[int, int]]:
    """Parse diff hunks and return inclusive line ranges in the new file."""
    if not patch:
        return []

    ranges: list[tuple[int, int]] = []
    for match in HUNK_PATTERN.finditer(patch):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            continue
        end = start + count - 1
        ranges.append((start, end))
    return ranges
