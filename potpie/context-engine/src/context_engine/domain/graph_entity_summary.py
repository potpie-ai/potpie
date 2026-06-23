"""Helpers for compact graph entity summaries."""

from __future__ import annotations

from typing import Any, Mapping

MAX_ENTITY_SUMMARY_CHARS = 320


def compact_entity_summary(
    *candidates: Any, max_chars: int = MAX_ENTITY_SUMMARY_CHARS
) -> str:
    """Return the first non-empty candidate as a compact one-line summary."""
    for candidate in candidates:
        text = _clean_text(candidate)
        if text:
            return _truncate(text, max_chars)
    return ""


def normalize_entity_properties(
    properties: Mapping[str, Any], *, entity_key: str
) -> dict[str, Any]:
    """Ensure entity property bags carry non-empty name, summary, and description."""
    props = dict(properties)
    name = _clean_text(props.get("name")) or entity_key
    props["name"] = name
    summary = compact_entity_summary(
        props.get("summary"),
        props.get("description"),
        props.get("title"),
        name,
        entity_key,
    )
    props["summary"] = summary
    if not _clean_text(props.get("description")):
        props["description"] = summary
    return props


def merge_entity_display_properties(
    properties: Mapping[str, Any],
    *,
    existing: Mapping[str, Any],
    entity_key: str,
) -> dict[str, Any]:
    """Merge an entity write into stored props without clobbering authored text.

    Authored incoming ``name`` / ``summary`` / ``description`` win; otherwise
    the existing non-empty stored value is kept; only a genuinely new node
    falls back to the entity key. Dict-backed stores use this so a bare
    re-reference (key + type only) never downgrades an authored summary to a
    key-derived one. The Cypher writer implements the same rule with CASE
    expressions.
    """
    props = dict(properties)
    authored_name = _clean_text(props.pop("name", None))
    authored_summary = compact_entity_summary(
        props.pop("summary", None),
        props.get("description"),
        props.get("title"),
        authored_name,
    )
    authored_description = _clean_text(props.pop("description", None)) or authored_summary
    out = dict(props)
    out["name"] = authored_name or _clean_text(existing.get("name")) or entity_key
    out["summary"] = (
        authored_summary or _clean_text(existing.get("summary")) or entity_key
    )
    out["description"] = (
        authored_description
        or _clean_text(existing.get("description"))
        or out["summary"]
    )
    return out


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 1].rstrip()
    return f"{trimmed}..."
