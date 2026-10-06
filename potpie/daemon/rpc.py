"""Local daemon RPC serialization helpers.

The daemon transport is deliberately thin: it moves calls to the in-daemon
``HostShell`` without teaching the CLI about every service DTO. Dataclasses keep
their module/class identity across the wire so command code can keep using
normal attribute access and local helper methods such as ``to_dict()``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

TYPE_KEY = "__potpie_rpc_type__"
_ALLOWED_CLASS_MODULE_PREFIXES = (
    "potpie_context_core.",
    "potpie_context_engine.domain.",
    "potpie.daemon.ports.",
)


def encode(value: Any, *, omit_defaults: bool = False) -> Any:
    """Encode common domain values into JSON-compatible data.

    ``omit_defaults`` leaves out every dataclass field still at its default.
    The decoder rebuilds objects with a strict ``cls(**raw)``, so a field the
    receiver's build does not have is fatal even when it carries the default —
    which made every added field, however optional, a break for an older peer
    (``ClaimQueryFilter.include_retired`` against a server that predates it).
    Requests are encoded this way: a call that does not use a new field sends
    exactly what an older client would, and one that does still fails loudly,
    because an older host cannot honour it. The receiver supplies its own
    default for an omitted field, so changing a wire field's default is a
    behaviour change for peers on the other side of that change.
    """
    if is_dataclass(value) and not isinstance(value, type):
        cls = value.__class__
        return {
            TYPE_KEY: "dataclass",
            "class": f"{cls.__module__}:{cls.__qualname__}",
            "value": {
                field.name: encode(item, omit_defaults=omit_defaults)
                for field, item in _wire_fields(value, omit_defaults=omit_defaults)
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
        return {
            TYPE_KEY: "tuple",
            "items": [encode(item, omit_defaults=omit_defaults) for item in value],
        }
    if isinstance(value, (list, set, frozenset)):
        return [encode(item, omit_defaults=omit_defaults) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): encode(item, omit_defaults=omit_defaults)
            for key, item in value.items()
        }
    return value


def _wire_fields(
    value: Any, *, omit_defaults: bool
) -> Iterator[tuple[Field[Any], Any]]:
    for field in fields(value):
        item = getattr(value, field.name)
        if omit_defaults and _is_default(field, item):
            continue
        yield field, item


def _is_default(field: Field[Any], item: Any) -> bool:
    """Is ``item`` this field's default? Unsure means no: the field is sent."""
    if field.default is not MISSING:
        default = field.default
    elif field.default_factory is not MISSING:
        default = field.default_factory()
    else:
        return False
    try:
        # The type check keeps ``0``/``0.0`` from passing for ``False``.
        return type(item) is type(default) and bool(item == default)
    except Exception:  # noqa: BLE001 - an odd __eq__ only costs sending the field
        return False


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
    if module_name == "potpie_context_core.graph_restore" or not module_name.startswith(
        _ALLOWED_CLASS_MODULE_PREFIXES
    ):
        raise TypeError(f"RPC class module not allowed: {module_name}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, type) and obj.__module__ == "potpie_context_core.graph_restore":
        raise TypeError("internal restore types cannot be decoded from RPC input")
    if not isinstance(obj, type):
        raise TypeError(f"RPC class reference is not a class: {ref}")
    return obj


__all__ = ["TYPE_KEY", "decode", "encode"]
