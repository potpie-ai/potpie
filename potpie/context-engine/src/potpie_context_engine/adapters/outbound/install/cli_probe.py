"""Bounded executable probe for the agent-facing Potpie workflow."""

from __future__ import annotations

import re
import subprocess
from typing import Any

REQUIRED_COMMAND_GROUPS: dict[tuple[str, ...], tuple[str, ...]] = {
    (): ("doctor", "graph", "record", "resolve", "resource", "skills", "status"),
    ("graph",): (
        "catalog",
        "commit",
        "mutation-template",
        "neighborhood",
        "propose",
        "read",
        "search-entities",
    ),
    ("resource",): ("get", "import"),
    ("skills",): ("install", "remove", "status", "update"),
}

REQUIRED_COMMANDS = tuple(
    " ".join((*group, command))
    for group, commands in REQUIRED_COMMAND_GROUPS.items()
    for command in commands
)


def probe_cli_surface(executable: str | None) -> dict[str, Any]:
    """Start the CLI and verify every command needed by the installed guidance."""
    if not executable:
        return {
            "ok": False,
            "commands": [],
            "missing_commands": list(REQUIRED_COMMANDS),
            "probes": [],
            "detail": "not on PATH",
        }

    found: list[str] = []
    probes: list[dict[str, Any]] = []
    for group, expected in REQUIRED_COMMAND_GROUPS.items():
        argv = [executable, *group, "--help"]
        try:
            proc = subprocess.run(  # noqa: S603 - executable is resolved from PATH
                argv,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            probes.append(
                {
                    "command": " ".join((*group, "--help")),
                    "ok": False,
                    "exit_code": None,
                    "detail": type(exc).__name__,
                }
            )
            continue
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        for command in expected:
            if _help_lists(output, command):
                found.append(" ".join((*group, command)))
        probes.append(
            {
                "command": " ".join((*group, "--help")),
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "detail": None if proc.returncode == 0 else "help command failed",
            }
        )

    missing = [command for command in REQUIRED_COMMANDS if command not in found]
    return {
        "ok": all(row["ok"] for row in probes) and not missing,
        "commands": found,
        "missing_commands": missing,
        "probes": probes,
        "detail": None if not missing else "required commands are unavailable",
    }


def _help_lists(output: str, command: str) -> bool:
    return bool(re.search(rf"(?m)^[^\w]*{re.escape(command)}(?:\s|$)", output))


__all__ = ["REQUIRED_COMMANDS", "REQUIRED_COMMAND_GROUPS", "probe_cli_surface"]
