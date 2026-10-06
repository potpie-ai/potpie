"""Explicit protocol projection shared by full JSON, compact cards and RPC."""

from collections.abc import Mapping
from typing import Any

_KEYS = frozenset(
    {
        "kind",
        "entity_key",
        "entity_type",
        "summary",
        "protocol",
        "message",
        "fields",
        "claims",
        "coverage",
        "coverage_status",
        "source_refs",
        "retrieval",
        "score",
    }
)


def protocol_read_item(payload: Mapping[str, Any], *, score: float) -> dict[str, Any]:
    return {
        **{key: value for key, value in payload.items() if key in _KEYS},
        "score": score,
    }


def protocol_item_for_detail(
    payload: Mapping[str, Any], *, detail: str
) -> dict[str, Any]:
    out = {key: value for key, value in payload.items() if key in _KEYS}
    if detail != "full":
        out.pop("fields", None)
        out.pop("claims", None)
        out["protocol"] = {
            key: value
            for key, value in out.get("protocol", {}).items()
            if key
            in {
                "name",
                "namespace",
                "identifier",
                "revision",
                "profile",
                "unresolved_source",
            }
        }
        out["message"] = {
            key: value
            for key, value in out.get("message", {}).items()
            if key in {"name", "kind", "direction", "discriminator", "protocol_key"}
        }
    return out
