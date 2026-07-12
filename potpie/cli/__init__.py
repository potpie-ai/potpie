"""Root-owned Potpie CLI package."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["host_cli"]


def __getattr__(name: str) -> ModuleType:
    if name == "host_cli":
        return import_module("potpie.cli.main")
    raise AttributeError(name)
