from __future__ import annotations

from collections.abc import Mapping
from typing import Final

FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def clean_env_value(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def env_value(environ: Mapping[str, str], name: str) -> str | None:
    return clean_env_value(environ.get(name))


def flag_value(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in FALSE_VALUES:
        return False
    if lowered in TRUE_VALUES:
        return True
    return bool(lowered)


def bool_env_value(value: bool) -> str:
    return "1" if value else "0"


__all__ = [
    "bool_env_value",
    "clean_env_value",
    "env_value",
    "flag_value",
]
