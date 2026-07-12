"""Static Potpie CLI command contract used by bundled-skill validation."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def load_command_manifest() -> dict[str, Any]:
    manifest = resources.files("potpie.skills").joinpath("command_manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1":
        raise RuntimeError("unsupported Potpie skill command manifest version")
    return data


def command_option_specs() -> dict[tuple[str, ...], frozenset[str]]:
    commands = load_command_manifest()["commands"]
    return {
        tuple(path.split(" ")): frozenset(str(option) for option in options)
        for path, options in commands.items()
    }


def root_options() -> frozenset[str]:
    return frozenset(str(option) for option in load_command_manifest()["root_options"])


__all__ = ["command_option_specs", "load_command_manifest", "root_options"]
