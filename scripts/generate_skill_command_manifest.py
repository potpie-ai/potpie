"""Generate the checked-in command manifest from root Typer registration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from typer.models import OptionInfo

from potpie.cli.main import app


def _command_name(command: Any) -> str:
    if command.name:
        return str(command.name)
    return str(command.callback.__name__).replace("_", "-")


def _options(callback: Any) -> list[str]:
    options: set[str] = set()
    for parameter in inspect.signature(callback).parameters.values():
        default = parameter.default
        if not isinstance(default, OptionInfo):
            continue
        for declaration in default.param_decls:
            for part in str(declaration).split("/"):
                if part.startswith("-"):
                    options.add(part)
    return sorted(options)


def collect_manifest() -> dict[str, object]:
    commands: dict[str, list[str]] = {}

    def collect(typer_app: Any, path: tuple[str, ...] = ()) -> None:
        for command in typer_app.registered_commands:
            if command.callback is None:
                continue
            command_path = (*path, _command_name(command))
            commands[" ".join(command_path)] = _options(command.callback)
        for group in typer_app.registered_groups:
            collect(group.typer_instance, (*path, str(group.name)))

    collect(app)
    callback = app.registered_callback
    root_options = (
        _options(callback.callback)
        if callback is not None and callback.callback is not None
        else []
    )
    return {
        "schema_version": "1",
        "root_options": root_options,
        "commands": dict(sorted(commands.items())),
    }


def main() -> None:
    output = Path(__file__).parents[1] / "potpie" / "skills" / "command_manifest.json"
    output.write_text(
        json.dumps(collect_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
