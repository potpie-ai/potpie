"""Generic plugin registry used for transports, components, and service backends."""

from __future__ import annotations
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class UnknownPlugin(KeyError):
    """Raised when create() is called with a name not in the registry."""


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._factories:
            raise ValueError(f"plugin already registered: {name!r}")
        self._factories[name] = factory

    def create(self, plugin_name: str, **cfg) -> T:
        try:
            factory = self._factories[plugin_name]
        except KeyError as exc:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise UnknownPlugin(
                f"unknown plugin {plugin_name!r}; known: {known}"
            ) from exc
        return factory(**cfg)

    def names(self) -> list[str]:
        return list(self._factories)
