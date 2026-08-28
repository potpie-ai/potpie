"""Transport-safe serialization helpers for public graph contracts."""

from __future__ import annotations

from collections import abc as collections_abc
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import types
from typing import (
    Any,
    Literal,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

T = TypeVar("T")


def to_transport(value: Any) -> Any:
    """Recursively convert a public contract value to JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_transport(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_transport(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
    if isinstance(value, Mapping):
        return {str(key): to_transport(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_transport(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not transport serializable")


def from_transport(contract: type[T], payload: Mapping[str, Any]) -> T:
    """Hydrate a public DTO that provides ``from_dict`` or ``parse``."""

    from_dict = getattr(contract, "from_dict", None)
    if callable(from_dict):
        return from_dict(payload)
    parse = getattr(contract, "parse", None)
    if callable(parse):
        return parse(payload)
    if is_dataclass(contract):
        hints = get_type_hints(contract)
        values = {
            item.name: _hydrate_value(hints.get(item.name, Any), payload[item.name])
            for item in fields(contract)
            if item.init and item.name in payload
        }
        return contract(**values)
    raise TypeError(f"{contract.__name__} does not support public hydration")


def _hydrate_value(annotation: Any, value: Any) -> Any:
    if annotation in {Any, object} or isinstance(annotation, TypeVar):
        return value
    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, types.UnionType}:
        candidates = tuple(
            candidate for candidate in args if candidate is not type(None)
        )
        structured = tuple(
            candidate
            for candidate in candidates
            if candidate not in {Any, object, str, int, float, bool}
        )
        for candidate in (*structured, *candidates):
            try:
                return _hydrate_value(candidate, value)
            except (TypeError, ValueError):
                continue
        return value
    if origin is Literal:
        return value
    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if annotation is date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return value if isinstance(value, annotation) else annotation(value)
    if origin in {list, collections_abc.Sequence}:
        item_type = args[0] if args else Any
        return [_hydrate_value(item_type, item) for item in value]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_hydrate_value(args[0], item) for item in value)
        return tuple(
            _hydrate_value(args[index], item) if index < len(args) else item
            for index, item in enumerate(value)
        )
    if origin in {set, frozenset}:
        item_type = args[0] if args else Any
        hydrated = (_hydrate_value(item_type, item) for item in value)
        return origin(hydrated)
    if origin in {
        dict,
        Mapping,
        collections_abc.Mapping,
        collections_abc.MutableMapping,
    }:
        key_type, item_type = args if len(args) == 2 else (Any, Any)
        return {
            _hydrate_value(key_type, key): _hydrate_value(item_type, item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"{annotation.__name__} requires an object payload")
        return from_transport(annotation, value)
    return value


__all__ = ["from_transport", "to_transport"]
