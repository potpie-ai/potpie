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

#: What a decoder does with a field the target dataclass does not declare.
#: Requests are decoded with ``"reject"``: a host must never silently drop part
#: of what a caller asked for. Results are decoded with ``"ignore"``, so an
#: older client keeps reading a newer host's answers.
UnknownFields = Literal["reject", "ignore"]


def decode_dataclass(
    model: type[T],
    value: Mapping[str, object],
    *,
    unknown: UnknownFields = "reject",
) -> T:
    """Reconstruct one exact dataclass; missing fields take their defaults.

    A field the dataclass does not declare is refused unless ``unknown`` is
    ``"ignore"``, which drops it, recursively through nested dataclasses.
    """

    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} is not a dataclass type")
    model_fields = {item.name: item for item in fields(model)}
    extra = set(value).difference(model_fields)
    if extra and unknown == "reject":
        raise TypeError(f"unexpected fields for {model.__name__}: {sorted(extra)!r}")
    hints = get_type_hints(model)
    decoded: dict[str, object] = {}
    for name, item in model_fields.items():
        if name in value:
            decoded[name] = decode_typed_value(
                value[name], hints[name], unknown=unknown
            )
        elif item.default is MISSING and item.default_factory is MISSING:
            raise TypeError(f"missing field for {model.__name__}: {name}")
    return model(**decoded)


def decode_typed_value(
    value: object, annotation: object, *, unknown: UnknownFields = "reject"
) -> object:
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
                return decode_typed_value(value, candidate, unknown=unknown)
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
        return tuple(
            decode_typed_value(item, item_type, unknown=unknown) for item in value
        )

    if origin in {list}:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        item_type = args[0] if args else Any
        return [decode_typed_value(item, item_type, unknown=unknown) for item in value]

    if origin in {dict, Mapping, MappingABC}:
        if not isinstance(value, MappingABC):
            raise TypeError("expected an object")
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        return {
            decode_typed_value(key, key_type, unknown=unknown): decode_typed_value(
                item, value_type, unknown=unknown
            )
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
            annotation,
            {str(key): item for key, item in value.items()},
            unknown=unknown,
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


def unknown_field_paths(model: type, value: Mapping[str, object]) -> tuple[str, ...]:
    """Dotted paths of every field in ``value`` that ``model`` does not declare.

    Walks nested dataclasses, sequences of them and optional ones, so a host can
    name exactly which part of a request it does not know (``"actor.team"``)
    instead of refusing the whole payload. A path is a schema path: items of a
    sequence share their field's path, and mapping values sit under ``*``.
    Values that do not match their declared shape at all are not reported here;
    decoding refuses those.
    """

    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} is not a dataclass type")
    return tuple(sorted(_unknown_paths(value, model, "")))


def _unknown_paths(value: object, annotation: object, prefix: str) -> set[str]:
    if annotation in {Any, object}:
        return set()
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        if value is None:
            return set()
        found: list[set[str]] = []
        for candidate in args:
            if candidate is type(None) or not _shape_matches(value, candidate):
                continue
            paths = _unknown_paths(value, candidate, prefix)
            if not paths:
                return set()
            found.append(paths)
        return min(found, key=len) if found else set()
    if origin in {tuple, list}:
        if not isinstance(value, (list, tuple)):
            return set()
        if origin is tuple and args and args[-1] is not Ellipsis:
            paths: set[str] = set()
            for item, item_type in zip(value, args):
                paths |= _unknown_paths(item, item_type, prefix)
            return paths
        item_type = args[0] if args else Any
        paths = set()
        for item in value:
            paths |= _unknown_paths(item, item_type, prefix)
        return paths
    if origin in {dict, Mapping, MappingABC}:
        if not isinstance(value, MappingABC) or len(args) < 2:
            return set()
        paths = set()
        for item in value.values():
            paths |= _unknown_paths(item, args[1], f"{prefix}*.")
        return paths
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, MappingABC):
            return set()
        hints = get_type_hints(annotation)
        declared = {item.name for item in fields(annotation)}
        paths = set()
        for key, item in value.items():
            name = str(key)
            if name not in declared:
                paths.add(f"{prefix}{name}")
            else:
                paths |= _unknown_paths(item, hints[name], f"{prefix}{name}.")
        return paths
    return set()


def _shape_matches(value: object, annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin in {tuple, list}:
        return isinstance(value, (list, tuple))
    if origin in {dict, Mapping, MappingABC}:
        return isinstance(value, MappingABC)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return isinstance(value, MappingABC)
    return False


def _plain_value(value: object) -> object:
    if isinstance(value, MappingABC):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


__all__ = [
    "UnknownFields",
    "decode_dataclass",
    "decode_typed_value",
    "unknown_field_paths",
]
