"""Bounded executable probe for the agent-facing Potpie workflow.

The installed skills tell agents to run ``potpie`` from ``PATH``. That
executable can be a different, older install than the one running ``setup``,
so the probe starts it and checks that every command the guidance names is
listed in its help.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

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

_HELP_TIMEOUT_S = 10
#: ``subprocess.CREATE_NO_WINDOW`` by value; the name exists only on Windows.
_WINDOWS_CREATE_NO_WINDOW: Final[int] = 0x08000000


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

    groups = list(REQUIRED_COMMAND_GROUPS.items())
    # One help process per group, run side by side: each pays the CLI's start-up
    # cost, and setup waits for all of them.
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = list(pool.map(lambda item: _probe_group(executable, *item), groups))

    found = {command for commands, _probe in results for command in commands}
    probes = [probe for _commands, probe in results]
    missing = [command for command in REQUIRED_COMMANDS if command not in found]
    return {
        "ok": all(row["ok"] for row in probes) and not missing,
        "commands": [command for command in REQUIRED_COMMANDS if command in found],
        "missing_commands": missing,
        "probes": probes,
        "detail": None if not missing else "required commands are unavailable",
    }


def _probe_group(
    executable: str, group: tuple[str, ...], expected: tuple[str, ...]
) -> tuple[list[str], dict[str, Any]]:
    argv = [executable, *group, "--help"]
    label = " ".join((*group, "--help"))
    try:
        # Help goes to a file, not a pipe: on Windows a timed-out child's
        # inherited pipe can keep ``communicate()`` waiting forever (see
        # ``domain.git_probe``), and no console window may open.
        with tempfile.TemporaryFile() as out:
            proc = subprocess.run(  # noqa: S603 - executable is resolved from PATH
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=_HELP_TIMEOUT_S,
                **_no_window_kwargs(),
            )
            out.seek(0)
            output = out.read().decode("utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], {
            "command": label,
            "ok": False,
            "exit_code": None,
            "detail": type(exc).__name__,
        }
    found = [
        " ".join((*group, command))
        for command in expected
        if _help_lists(output, command)
    ]
    return found, {
        "command": label,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "detail": None if proc.returncode == 0 else "help command failed",
    }


def _no_window_kwargs() -> dict[str, int]:
    if os.name == "nt":
        return {"creationflags": _WINDOWS_CREATE_NO_WINDOW}
    return {}


def _help_lists(output: str, command: str) -> bool:
    return bool(re.search(rf"(?m)^[^\w]*{re.escape(command)}(?:\s|$)", output))


__all__ = ["REQUIRED_COMMANDS", "REQUIRED_COMMAND_GROUPS", "probe_cli_surface"]
