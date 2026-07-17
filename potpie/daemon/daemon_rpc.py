"""Local daemon RPC serialization helpers.

The daemon transport is deliberately thin: it moves calls to the in-daemon
``HostShell`` without teaching the CLI about every service DTO. Dataclasses keep
their module/class identity across the wire so command code can keep using
normal attribute access and local helper methods such as ``to_dict()``.
"""

from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

TYPE_KEY = "__potpie_rpc_type__"
_ALLOWED_CLASS_MODULE_PREFIXES = (
    "potpie_context_core.",
    "potpie_context_engine.domain.",
    "potpie.daemon.ports.",
)


def encode(value: Any) -> Any:
    """Encode common domain values into JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        cls = value.__class__
        return {
            TYPE_KEY: "dataclass",
            "class": f"{cls.__module__}:{cls.__qualname__}",
            "value": {
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, datetime):
        return {TYPE_KEY: "datetime", "value": value.isoformat()}
    if isinstance(value, Enum):
        cls = value.__class__
        return {
            TYPE_KEY: "enum",
            "class": f"{cls.__module__}:{cls.__qualname__}",
            "value": value.value,
        }
    if isinstance(value, Path):
        return {TYPE_KEY: "path", "value": str(value)}
    if isinstance(value, tuple):
        return {TYPE_KEY: "tuple", "items": [encode(item) for item in value]}
    if isinstance(value, (list, set, frozenset)):
        return [encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    return value


def decode(value: Any) -> Any:
    """Decode values produced by :func:`encode`."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get(TYPE_KEY)
    if marker == "dataclass":
        cls = _load_class(value["class"])
        raw = value.get("value") or {}
        return cls(**{key: decode(item) for key, item in raw.items()})
    if marker == "datetime":
        return datetime.fromisoformat(value["value"])
    if marker == "enum":
        cls = _load_class(value["class"])
        return cls(value["value"])
    if marker == "path":
        return Path(value["value"])
    if marker == "tuple":
        return tuple(decode(item) for item in value.get("items") or [])

    return {key: decode(item) for key, item in value.items()}


def _load_class(ref: str) -> type:
    module_name, qualname = ref.split(":", 1)
    if not module_name.startswith(_ALLOWED_CLASS_MODULE_PREFIXES):
        raise TypeError(f"RPC class module not allowed: {module_name}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"RPC class reference is not a class: {ref}")
    return obj


__all__ = ["TYPE_KEY", "decode", "encode"]
