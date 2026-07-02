"""Local daemon RPC serialization helpers.

The daemon transport is deliberately thin: it moves calls to the in-daemon
``HostShell`` without teaching the CLI about every service DTO. Dataclasses keep
their module/class identity across the wire so command code can keep using
normal attribute access and local helper methods such as ``to_dict()``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from domain.ports.daemon.rpc_contract import (
    RPC_SURFACES,
    RpcSurfaceSpec,
    assert_rpc_class_allowed,
    class_ref,
    load_rpc_class,
)

TYPE_KEY = "__potpie_rpc_type__"


def rpc_child_surface(surface: str, name: str) -> str | None:
    """Return the nested RPC surface for ``surface.name`` when exposed."""
    return _surface_spec(surface).children.get(name)


def is_rpc_method_allowed(surface: str, method: str) -> bool:
    """Return whether ``surface.method`` is part of the daemon RPC contract."""
    return method in _surface_spec(surface).methods


def is_rpc_attr_allowed(surface: str, name: str) -> bool:
    """Return whether ``surface.name`` is an exposed remote attribute."""
    return name in _surface_spec(surface).attrs


def validate_rpc_method(surface: str, method: str) -> None:
    """Fail closed when a caller asks for a method outside the RPC contract."""
    if not is_rpc_method_allowed(surface, method):
        raise ValueError(f"invalid RPC method: {surface}.{method}")


def validate_rpc_attr(surface: str, name: str) -> None:
    """Fail closed when a caller asks for an attribute outside the RPC contract."""
    if not is_rpc_attr_allowed(surface, name):
        raise ValueError(f"invalid RPC attribute: {surface}.{name}")


def _surface_spec(surface: str) -> RpcSurfaceSpec:
    try:
        return RPC_SURFACES[surface]
    except KeyError as exc:
        raise ValueError(f"invalid RPC surface: {surface}") from exc


def encode(value: Any) -> Any:
    """Encode common domain values into JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        cls = value.__class__
        assert_rpc_class_allowed(cls)
        return {
            TYPE_KEY: "dataclass",
            "class": class_ref(cls),
            "value": {
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, datetime):
        return {TYPE_KEY: "datetime", "value": value.isoformat()}
    if isinstance(value, Enum):
        enum_cls = value.__class__
        assert_rpc_class_allowed(enum_cls)
        return {
            TYPE_KEY: "enum",
            "class": class_ref(enum_cls),
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
        cls = load_rpc_class(value["class"])
        raw = value.get("value") or {}
        return cls(**{key: decode(item) for key, item in raw.items()})
    if marker == "datetime":
        return datetime.fromisoformat(value["value"])
    if marker == "enum":
        cls = load_rpc_class(value["class"])
        return cls(value["value"])
    if marker == "path":
        return Path(value["value"])
    if marker == "tuple":
        return tuple(decode(item) for item in value.get("items") or [])

    return {key: decode(item) for key, item in value.items()}


__all__ = [
    "RPC_SURFACES",
    "TYPE_KEY",
    "RpcSurfaceSpec",
    "decode",
    "encode",
    "is_rpc_attr_allowed",
    "is_rpc_method_allowed",
    "rpc_child_surface",
    "validate_rpc_attr",
    "validate_rpc_method",
]
