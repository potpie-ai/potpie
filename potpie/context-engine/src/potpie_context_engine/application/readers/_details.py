"""Bounded, kind-specific entity details for graph reader payloads."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from potpie_context_core.ports.claim_query import ClaimQueryPort, entity_properties_many

_MAX_TEXT = 2_000
_MAX_LIST_ITEMS = 12


def entity_details(
    query: ClaimQueryPort,
    *,
    pot_id: str,
    entity_keys: Iterable[str],
    fields: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return public, bounded properties for a set of entities.

    The omitted map makes loss explicit without copying provenance and other
    internal node properties into every result. Entity neighborhoods remain the
    escape hatch for the complete stored value.
    """
    selected = tuple(dict.fromkeys(fields))
    properties = entity_properties_many(query, pot_id=pot_id, entity_keys=entity_keys)
    return {
        key: _bounded_properties(value, fields=selected)
        for key, value in properties.items()
        if value
    }


def _bounded_properties(
    properties: Mapping[str, Any], *, fields: Iterable[str]
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    omitted: dict[str, Any] = {}
    for field in fields:
        value = properties.get(field)
        if value is None or value == "" or value == [] or value == ():
            continue
        if isinstance(value, str):
            if len(value) > _MAX_TEXT:
                details[field] = value[:_MAX_TEXT]
                omitted[field] = {"characters": len(value) - _MAX_TEXT}
            else:
                details[field] = value
            continue
        if isinstance(value, (list, tuple)):
            bounded_values: list[Any] = []
            truncated_entries: dict[str, Any] = {}
            for index, entry in enumerate(value[:_MAX_LIST_ITEMS]):
                bounded, entry_omitted = _bounded_value(entry)
                bounded_values.append(bounded)
                if entry_omitted:
                    truncated_entries[str(index)] = entry_omitted
            details[field] = bounded_values
            if len(value) > _MAX_LIST_ITEMS:
                omitted[field] = {"items": len(value) - _MAX_LIST_ITEMS}
            if truncated_entries:
                omitted.setdefault(field, {})["entries"] = truncated_entries
            continue
        if isinstance(value, (bool, int, float)):
            details[field] = value
    if omitted:
        details["omitted"] = omitted
    return details


def _bounded_value(value: Any) -> tuple[Any, Mapping[str, Any] | None]:
    if isinstance(value, str):
        if len(value) <= _MAX_TEXT:
            return value, None
        return value[:_MAX_TEXT], {"characters": len(value) - _MAX_TEXT}
    if isinstance(value, (bool, int, float)) or value is None:
        return value, None
    # Entity detail fields are expected to be scalars or lists of scalars.
    # Stringify any legacy nested value into the same bounded text budget
    # rather than recursively exposing an arbitrary internal payload.
    rendered = str(value)
    if len(rendered) <= _MAX_TEXT:
        return rendered, {"coerced": True}
    return rendered[:_MAX_TEXT], {
        "coerced": True,
        "characters": len(rendered) - _MAX_TEXT,
    }


__all__ = ["entity_details"]
