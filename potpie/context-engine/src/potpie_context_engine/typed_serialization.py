"""Dependency-light reconstruction for typed request and result dataclasses."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import UnionType
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
from uuid import UUID


T = TypeVar("T")


def decode_dataclass(model: type[T], value: Mapping[str, object]) -> T:
    """Reconstruct one exact dataclass and reject unknown or missing fields."""

    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} is not a dataclass type")
    model_fields = {item.name: item for item in fields(model)}
    unknown = set(value).difference(model_fields)
    if unknown:
        raise TypeError(f"unexpected fields for {model.__name__}: {sorted(unknown)!r}")
    hints = get_type_hints(model)
    decoded: dict[str, object] = {}
    for name, item in model_fields.items():
        if name in value:
            decoded[name] = decode_typed_value(value[name], hints[name])
        elif item.default is MISSING and item.default_factory is MISSING:
            raise TypeError(f"missing field for {model.__name__}: {name}")
    return model(**decoded)


def decode_typed_value(value: object, annotation: object) -> object:
    """Decode JSON-shaped data according to a dependency-light type hint."""

    if annotation in {Any, object}:
        return _plain_value(value)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {UnionType, Union}:
        if value is None and type(None) in args:
            return None
        failures: list[Exception] = []
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return decode_typed_value(value, candidate)
            except (TypeError, ValueError) as exc:
                failures.append(exc)
        raise TypeError("value does not match any allowed type") from (
            failures[-1] if failures else None
        )

    if origin is Literal:
        if value not in args:
            raise ValueError(f"expected one of {args!r}")
        return value

    if origin in {tuple}:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        item_type = args[0] if args else Any
        return tuple(decode_typed_value(item, item_type) for item in value)

    if origin in {list}:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        item_type = args[0] if args else Any
        return [decode_typed_value(item, item_type) for item in value]

    if origin in {dict, Mapping, MappingABC}:
        if not isinstance(value, MappingABC):
            raise TypeError("expected an object")
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        return {
            decode_typed_value(key, key_type): decode_typed_value(item, value_type)
            for key, item in value.items()
        }

    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise TypeError("expected an ISO datetime")
        return datetime.fromisoformat(value)
    if annotation is date:
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            raise TypeError("expected an ISO date")
        return date.fromisoformat(value)
    if annotation is Path:
        if not isinstance(value, str):
            raise TypeError("expected a path string")
        return Path(value)
    if annotation is UUID:
        if not isinstance(value, str):
            raise TypeError("expected a UUID string")
        return UUID(value)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, MappingABC):
            raise TypeError(f"expected an object for {annotation.__name__}")
        return decode_dataclass(
            annotation, {str(key): item for key, item in value.items()}
        )

    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("expected a boolean")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("expected an integer")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("expected a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("expected a string")
        return value
    if value is None:
        raise TypeError("null is not allowed")
    return value


def _plain_value(value: object) -> object:
    if isinstance(value, MappingABC):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


__all__ = ["decode_dataclass", "decode_typed_value"]
