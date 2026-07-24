"""Diagnostics for how the ``potpie`` CLI is installed on the host."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

CLI_TOOL_NAME = "potpie-context-engine"
CLI_EXECUTABLE = "potpie"
_UV_TOOL_NAMES = frozenset({"potpie", "potpie-context-engine", "context-engine"})

_DIAGNOSTIC_COMMANDS = (
    "uv tool list",
    "which -a potpie",
    'head -n 1 "$(command -v potpie)"',
    "make cli-status",
    "make cli-install",
)

_LOCAL_REINSTALL_HINT = (
    "Check with `make cli-status` or `potpie doctor`. "
    "Repo-local reinstall: `make cli-install` "
    "(builds UI, stops old daemon) — not raw `uv tool install`."
)
_PUBLISHED_HINT = (
    "Check with `make cli-status` or `potpie doctor`. "
    "Published-package reinstall: `uv tool install potpie` "
    "(or `pip install potpie`)."
)


def collect_cli_install_status() -> dict[str, Any]:
    """Return install facts for ``potpie doctor`` and ``make cli-status``."""
    paths_on_path = _potpie_paths_on_path()
    primary_path = paths_on_path[0] if paths_on_path else None
    python_interpreter = _python_from_script(primary_path) if primary_path else None
    python_version = _python_version(python_interpreter)

    listed_uv = _uv_tool_list_status()
    active_uv = _active_uv_tool_from_executable(primary_path)
    via_uv_tool = bool(active_uv and active_uv.get("tool_name") in _UV_TOOL_NAMES)
    editable = bool(via_uv_tool and _is_editable_uv_tool(active_uv["tool_root"]))

    package_version = _package_version_via_interpreter(python_interpreter)
    if package_version is None:
        package_version = _installed_package_version()

    uv_tool_version = None
    if via_uv_tool:
        uv_tool_version = listed_uv.get("versions", {}).get(
            str(active_uv["tool_name"])
        ) or listed_uv.get("version")

    hint = None
    if via_uv_tool and editable:
        hint = _LOCAL_REINSTALL_HINT
    elif via_uv_tool:
        hint = _PUBLISHED_HINT

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
        # True when *some* potpie-related uv tool is listed (may not back PATH).
        "uv_tool_installed": bool(listed_uv.get("installed")),
        "uv_tool_version": uv_tool_version or listed_uv.get("version"),
        "uv_tool_name": active_uv.get("tool_name") if via_uv_tool else None,
        "uv_tool_root": str(active_uv["tool_root"]) if via_uv_tool else None,
        "editable": editable if via_uv_tool else None,
        # Only when the active PATH executable is backed by a uv tools env.
        "install_method": "uv_tool" if via_uv_tool else None,
        "diagnostic_commands": list(_DIAGNOSTIC_COMMANDS),
        "hint": hint,
        "pip_show_note": (
            "Do not use `python -m pip show potpie-context-engine` for local dev "
            "installs: `python` may be absent from PATH and the package lives in "
            "the uv tool environment. Prefer `uv tool list`, `which -a potpie`, "
            "`make cli-status`, and `make cli-install` for repo-local reinstalls."
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
    if status.get("editable"):
        parts.append("editable=true")
    if py:
        parts.append(f"python={py}")
    line = " ".join(parts)
    if status.get("editable"):
        line += " | tip: make cli-status (local reinstall: make cli-install)"
    elif via == "uv_tool":
        line += " | tip: make cli-status (published: uv tool install potpie)"
    return line


def _installed_package_version() -> str | None:
    try:
        return version(CLI_TOOL_NAME)
    except PackageNotFoundError:
        return None


def _package_version_via_interpreter(interpreter: str | None) -> str | None:
    """Read package version from the active CLI interpreter, not this process."""
    if not interpreter:
        return None
    for pkg in (CLI_TOOL_NAME, "potpie"):
        try:
            proc = subprocess.run(
                [
                    interpreter,
                    "-c",
                    (
                        "from importlib.metadata import version; "
                        f"print(version({pkg!r}))"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            value = (proc.stdout or "").strip()
            if value:
                return value
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
    try:
        proc = subprocess.run(
            [interpreter, "--version"],
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


def _active_uv_tool_from_executable(script_path: str | None) -> dict[str, Any] | None:
    """Return the uv tools env that backs ``script_path``, if any."""
    if not script_path:
        return None
    try:
        resolved = Path(script_path).resolve()
    except OSError:
        return None
    parts = resolved.parts
    for i in range(len(parts) - 2):
        if parts[i] == "uv" and parts[i + 1] == "tools":
            tool_name = parts[i + 2]
            tool_root = Path(*parts[: i + 3])
            return {"tool_name": tool_name, "tool_root": tool_root}
    return None


def _is_editable_uv_tool(tool_root: Path) -> bool:
    receipt = tool_root / "uv-receipt.toml"
    if receipt.is_file():
        try:
            text = receipt.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if re.search(r"\beditable\s*=", text):
            return True
    lib = tool_root / "lib"
    if not lib.is_dir():
        return False
    try:
        candidates = lib.rglob("direct_url.json")
    except OSError:
        return False
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("dir_info", {}).get("editable"):
            return True
    return False


def _uv_tool_list_status() -> dict[str, Any]:
    """Parse ``uv tool list`` for potpie-related tools (may not back PATH)."""
    if shutil.which("uv") is None:
        return {"installed": False, "version": None, "versions": {}}
    try:
        proc = subprocess.run(
            ["uv", "tool", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": False, "version": None, "versions": {}}
    if proc.returncode != 0:
        return {"installed": False, "version": None, "versions": {}}

    versions: dict[str, str | None] = {}
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        name, _, ver = stripped.partition(" ")
        if name in _UV_TOOL_NAMES:
            versions[name] = ver.removeprefix("v") or None

    if not versions:
        return {"installed": False, "version": None, "versions": {}}
    # Prefer the modern tool name when reporting a single version.
    preferred = (
        versions.get("potpie")
        or versions.get("potpie-context-engine")
        or versions.get("context-engine")
    )
    return {"installed": True, "version": preferred, "versions": versions}


def _uv_tool_status() -> dict[str, Any]:
    """Backward-compatible helper used by older callers/tests."""
    status = _uv_tool_list_status()
    return {"installed": status["installed"], "version": status["version"]}
