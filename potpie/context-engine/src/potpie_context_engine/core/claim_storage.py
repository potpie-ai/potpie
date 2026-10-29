"""Preserve raw embedded claim fields until an explicit write changes them.

This is read evidence, not a graph property. Portable snapshots and journals
use their existing schemas; only embedded storage and raw inventory overlay it.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StoredClaimSource:
    fields: Mapping[str, Any]
    hydrated_fields: Mapping[str, Any]


def stored_claim_record_id(
    fields: Mapping[str, Any], properties: Mapping[str, Any]
) -> str:
    """Tolerant read identity; the raw inventory separately diagnoses ambiguity."""
    for name in ("record_id", "uuid"):
        for namespace in (fields, properties):
            value = namespace.get(name)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def preserve_claim_source(
    current: Mapping[str, Any],
    source: StoredClaimSource | None,
) -> dict[str, Any]:
    result = deepcopy(dict(current))
    if source is None:
        return result
    # A separate presence test keeps absent, explicit null and invalid-but-parsed
    # as null distinct. Unknown legacy fields, including uuid, remain intact.
    for key in source.fields.keys() | source.hydrated_fields.keys():
        unchanged = (key in current) == (key in source.hydrated_fields) and (
            current.get(key) == source.hydrated_fields.get(key)
        )
        if unchanged:
            if key in source.fields:
                result[key] = deepcopy(source.fields[key])
            else:
                result.pop(key, None)
    return result
