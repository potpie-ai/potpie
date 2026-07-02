"""Diagnostics for how the ``potpie`` CLI is installed on the host."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

CLI_TOOL_NAME = "potpie-context-engine"
CLI_EXECUTABLE = "potpie"

_DIAGNOSTIC_COMMANDS = (
    "uv tool list",
    "which -a potpie",
    'head -n 1 "$(command -v potpie)"',
    "make cli-status",
)


def collect_cli_install_status() -> dict[str, Any]:
    """Return install facts for ``potpie doctor`` and ``make cli-status``."""
    paths_on_path = _potpie_paths_on_path()
    primary_path = paths_on_path[0] if paths_on_path else None
    python_interpreter = _python_from_script(primary_path) if primary_path else None
    python_version = _python_version(python_interpreter)
    uv_tool = _uv_tool_status()
    package_version = _installed_package_version()

    return {
        "package_name": CLI_TOOL_NAME,
        "package_version": package_version,
        "on_path": bool(paths_on_path),
        "paths": paths_on_path,
        "primary_path": primary_path,
        "python_interpreter": python_interpreter,
        "python_version": python_version,
        "runtime_python": sys.executable,
        "runtime_python_version": ".".join(map(str, sys.version_info[:3])),
        "uv_available": shutil.which("uv") is not None,
        "uv_tool_installed": uv_tool.get("installed"),
        "uv_tool_version": uv_tool.get("version"),
        "install_method": "uv_tool" if uv_tool.get("installed") else None,
        "diagnostic_commands": list(_DIAGNOSTIC_COMMANDS),
        "pip_show_note": (
            "Do not use `python -m pip show potpie-context-engine` for local dev "
            "installs: `python` may be absent from PATH and the package lives in "
            "the uv tool environment. Prefer `uv tool list`, `which -a potpie`, "
            "and `make cli-status`."
        ),
    }


def cli_install_human(status: dict[str, Any]) -> str:
    if not status.get("on_path"):
        return "cli: potpie NOT on PATH (run: make cli-install)"
    pkg = str(status.get("package_name") or CLI_TOOL_NAME)
    ver = status.get("package_version") or status.get("uv_tool_version") or "unknown"
    path = status.get("primary_path") or "unknown"
    py = status.get("python_version")
    via = status.get("install_method")
    parts = [f"cli: {pkg} {ver}", f"path={path}"]
    if via:
        parts.append(f"via={via}")
    if py:
        parts.append(f"python={py}")
    return " ".join(parts)


def _installed_package_version() -> str | None:
    try:
        return version(CLI_TOOL_NAME)
    except PackageNotFoundError:
        return None


def _potpie_paths_on_path() -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, CLI_EXECUTABLE)
        if not (os.path.isfile(candidate) or os.path.islink(candidate)):
            continue
        resolved = os.path.realpath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(candidate)
    return paths


def _python_from_script(script_path: str | None) -> str | None:
    if not script_path:
        return None
    try:
        with open(script_path, encoding="utf-8") as handle:
            first = handle.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None
    if first.startswith("#!"):
        return first[2:].strip() or None
    return None


def _python_version(interpreter: str | None) -> str | None:
    if not interpreter:
        return None
    args = shlex.split(interpreter)
    if not args:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - interpreter comes from installed CLI shebang.
            [*args, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (proc.stdout or proc.stderr or "").strip()
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
    return match.group(1) if match else output or None


def _uv_tool_status() -> dict[str, Any]:
    uv_path = shutil.which("uv")
    if uv_path is None:
        return {"installed": False, "version": None}
    try:
        proc = subprocess.run(  # noqa: S603 - resolved via PATH for install diagnostics.
            [uv_path, "tool", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": False, "version": None}
    if proc.returncode != 0:
        return {"installed": False, "version": None}
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        name, _, ver = stripped.partition(" ")
        if name == CLI_TOOL_NAME:
            return {"installed": True, "version": ver.removeprefix("v") or None}
    return {"installed": False, "version": None}
