"""Transport-neutral result normalization for local/daemon parity tests.

This belongs to test support, not the product contract. Later migration commits
can compare typed in-process results with decoded daemon results without making
transport correlation fields part of behavioral equality.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Mapping

TRANSPORT_ONLY_KEYS = frozenset(
    {
        "protocol_version",
        "request_id",
        "runtime_mode",
        "transport_metadata",
    }
)


def normalize_engine_result(
    value: Any,
    *,
    ignored_keys: Collection[str] = TRANSPORT_ONLY_KEYS,
) -> Any:
    """Return a deterministic JSON-compatible representation of a result.

    Dataclasses and Pydantic models are normalized structurally. Mapping keys
    are sorted, sets become deterministically ordered lists, and correlation
    fields that exist only on the daemon transport are omitted recursively.
    """

    ignored = frozenset(ignored_keys)
    return _normalize(value, ignored_keys=ignored)


def _normalize(value: Any, *, ignored_keys: frozenset[str]) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    elif hasattr(value, "model_dump") and callable(value.model_dump):
        value = value.model_dump(mode="python")

    if isinstance(value, Enum):
        return _normalize(value.value, ignored_keys=ignored_keys)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, ignored_keys=ignored_keys)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in ignored_keys
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item, ignored_keys=ignored_keys) for item in value]
        return sorted(normalized, key=_stable_sort_key)
    if isinstance(value, (list, tuple)):
        return [_normalize(item, ignored_keys=ignored_keys) for item in value]
    return value


def _stable_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


__all__ = ["TRANSPORT_ONLY_KEYS", "normalize_engine_result"]
