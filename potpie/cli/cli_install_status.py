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

# Diagnostics that ship with the wheel must only name commands the wheel
# provides. `make` targets exist in the source checkout, never in an install,
# so they are added only when the running CLI is an editable repo install.
_DIAGNOSTIC_COMMANDS = (
    "potpie doctor",
    "uv tool list",
    "which -a potpie",
)
_EDITABLE_DIAGNOSTIC_COMMANDS = ("make cli-status", "make cli-install")

_LOCAL_REINSTALL_HINT = (
    "Check with `potpie doctor`. Repo-local reinstall: `make cli-install` "
    "(builds UI, stops old daemon) — not raw `uv tool install`."
)
_PUBLISHED_HINT = (
    "Check with `potpie doctor`. Published-package reinstall: "
    "`uv tool install --force potpie` (or `pip install --upgrade potpie`)."
)


def collect_cli_install_status() -> dict[str, Any]:
    """Return install facts about **the running** ``potpie``.

    Resolution starts at this process (``sys.argv[0]`` / ``sys.executable``)
    rather than at ``$PATH``: scanning ``$PATH`` first made ``doctor`` report a
    *different* install's version, and made the running binary declare itself
    "NOT on PATH" whenever ``$PATH`` did not happen to contain it.
    """
    paths_on_path = _potpie_paths_on_path()
    running_path = _running_cli_path()
    # The running entry point is authoritative; PATH entries are context.
    primary_path = running_path or (paths_on_path[0] if paths_on_path else None)
    resolved_primary = _realpath(primary_path)
    other_paths = [
        path for path in paths_on_path if _realpath(path) != resolved_primary
    ]
    is_running = bool(running_path) and primary_path == running_path

    if is_running:
        # We are the process being diagnosed: ask ourselves, never a shebang.
        python_interpreter = sys.executable
        python_version = ".".join(map(str, sys.version_info[:3]))
    else:
        python_interpreter = _python_from_script(primary_path)
        python_version = _python_version(python_interpreter)

    listed_uv = _uv_tool_list_status()
    active_uv = _active_uv_tool_from_executable(primary_path)
    via_uv_tool = bool(active_uv and active_uv.get("tool_name") in _UV_TOOL_NAMES)
    editable = bool(
        via_uv_tool
        and _is_editable_uv_tool(active_uv["tool_root"], str(active_uv["tool_name"]))
    )

    package_version = (
        _installed_package_version()
        if is_running
        else _package_version_via_interpreter(python_interpreter)
    )
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

    diagnostics = list(_DIAGNOSTIC_COMMANDS)
    if editable:
        diagnostics.extend(_EDITABLE_DIAGNOSTIC_COMMANDS)

    return {
        "package_name": CLI_TOOL_NAME,
        "package_version": package_version,
        "product_version": _product_version(),
        "on_path": bool(paths_on_path),
        "paths": paths_on_path,
        "primary_path": primary_path,
        "running_path": running_path,
        # The running CLI is on PATH only if some PATH entry resolves to it.
        "running_on_path": bool(
            resolved_primary
            and any(_realpath(path) == resolved_primary for path in paths_on_path)
        ),
        "other_paths": other_paths,
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
        "diagnostic_commands": diagnostics,
        "hint": hint,
        "pip_show_note": (
            "Do not use `python -m pip show potpie-context-engine`: `python` may "
            "be absent from PATH and the package lives in the uv tool "
            "environment. Prefer `potpie doctor`, `uv tool list`, and "
            "`which -a potpie`."
        ),
    }


def _realpath(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def _product_version() -> str | None:
    try:
        return version("potpie")
    except PackageNotFoundError:
        return None


def _running_cli_path() -> str | None:
    """Path of the console script that started this process, if it is one.

    ``python -m potpie.cli.main`` and test harnesses do not run a console
    script, so this returns ``None`` there and PATH scanning takes over.
    """
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    if not argv0:
        return None
    name = os.path.basename(argv0)
    if name not in {CLI_EXECUTABLE, f"{CLI_EXECUTABLE}.exe"}:
        return None
    candidate = argv0
    if os.path.sep not in argv0:
        found = shutil.which(argv0)
        if not found:
            return None
        candidate = found
    if not os.path.isfile(candidate):
        return None
    return os.path.abspath(candidate)


def cli_install_human(status: dict[str, Any]) -> str:
    path = status.get("primary_path")
    if not path:
        return (
            "cli: potpie entry point not resolvable "
            "(install with: uv tool install potpie)"
        )
    pkg = str(status.get("package_name") or CLI_TOOL_NAME)
    ver = status.get("package_version") or status.get("uv_tool_version") or "unknown"
    product = status.get("product_version")
    py = status.get("python_version")
    via = status.get("install_method")
    parts = [f"cli: potpie {product}" if product else "cli: potpie", f"({pkg} {ver})"]
    parts.append(f"path={path}")
    if via:
        parts.append(f"via={via}")
    if status.get("editable"):
        parts.append("editable=true")
    if py:
        parts.append(f"python={py}")
    lines = [" ".join(parts)]
    if not status.get("running_on_path"):
        lines.append(
            "  note: this potpie is not the one on $PATH "
            "(run `uv tool update-shell`, or restart your shell)"
        )
    others = status.get("other_paths") or []
    if others:
        lines.append(f"  other installs on $PATH: {', '.join(others)}")
    if status.get("editable"):
        lines.append("  tip: local reinstall with `make cli-install`")
    elif via == "uv_tool":
        lines.append("  tip: reinstall with `uv tool install --force potpie`")
    return "\n".join(lines)


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
    """Interpreter named by a script's shebang, only if it *is* a Python.

    uv's tool-env entry points are ``#!/bin/sh`` wrappers. Running
    ``/bin/sh --version`` and scraping a number reported macOS bash's 3.2.57 as
    the CLI's Python version, so a non-Python shebang yields nothing here.
    """
    if not script_path:
        return None
    try:
        with open(script_path, encoding="utf-8") as handle:
            first = handle.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not first.startswith("#!"):
        return None
    shebang = first[2:].strip()
    if not shebang:
        return None
    interpreter = shebang.split()[0]
    # `#!/usr/bin/env python3.12` names the interpreter in the second word.
    if os.path.basename(interpreter) == "env":
        parts = shebang.split()
        interpreter = parts[1] if len(parts) > 1 else ""
    return interpreter if _is_python_interpreter(interpreter) else None


def _is_python_interpreter(interpreter: str) -> bool:
    return os.path.basename(interpreter or "").lower().startswith("python")


_PYTHON_VERSION_RE = re.compile(r"^Python (\d+\.\d+(?:\.\d+)?)", re.MULTILINE)


def _python_version(interpreter: str | None) -> str | None:
    if not interpreter or not _is_python_interpreter(interpreter):
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
    # Anchored on Python's own banner so no other tool's version can match.
    match = _PYTHON_VERSION_RE.search(output)
    return match.group(1) if match else None


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


def _package_names_for_tool(tool_name: str) -> frozenset[str]:
    """Distribution / requirement names that identify the tool package itself."""
    return frozenset(
        {
            tool_name,
            tool_name.replace("-", "_"),
            tool_name.replace("_", "-"),
        }
    )


def _is_editable_uv_tool(tool_root: Path, tool_name: str) -> bool:
    """Return True only when *this* tool package is an editable install."""
    package_names = _package_names_for_tool(tool_name)
    receipt = tool_root / "uv-receipt.toml"
    if receipt.is_file():
        try:
            text = receipt.read_text(encoding="utf-8")
        except OSError:
            text = ""
        receipt_editable = _receipt_tool_editable(text, package_names)
        if receipt_editable is not None:
            return receipt_editable

    return _direct_url_tool_editable(tool_root, package_names)


def _receipt_tool_editable(text: str, package_names: frozenset[str]) -> bool | None:
    """Parse uv-receipt.toml for the matching tool requirement's editable flag.

    Returns True/False when the tool requirement is found, else None.
    """
    requirements: list[Any] = []
    try:
        import tomllib

        data = tomllib.loads(text)
        tool = data.get("tool")
        if isinstance(tool, dict):
            raw = tool.get("requirements")
            if isinstance(raw, list):
                requirements = raw
        if not requirements:
            raw = data.get("requirements")
            if isinstance(raw, list):
                requirements = raw
    except Exception:
        requirements = []

    if requirements:
        for req in requirements:
            if not isinstance(req, dict):
                continue
            name = str(req.get("name") or "").strip()
            if name not in package_names:
                continue
            return bool(req.get("editable"))
        return None

    # Fallback: only treat editable as True when it appears on the matching req.
    for name in package_names:
        if re.search(
            rf'\{{\s*name\s*=\s*"{re.escape(name)}"\s*,[^{{}}]*\beditable\s*=',
            text,
        ):
            return True
        if re.search(rf'\{{\s*name\s*=\s*"{re.escape(name)}"', text):
            return False
    return None


def _direct_url_tool_editable(tool_root: Path, package_names: frozenset[str]) -> bool:
    """Check direct_url.json only under this tool's own dist-info directories."""
    lib = tool_root / "lib"
    if not lib.is_dir():
        return False

    try:
        site_package_paths = lib.rglob("site-packages")
    except OSError:
        return False

    site_packages: list[Path] = []
    try:
        for path in site_package_paths:
            try:
                if path.is_dir():
                    site_packages.append(path)
            except OSError:
                continue
    except OSError:
        # OSError can also arise while the rglob generator walks the tree.
        return False

    for site in site_packages:
        try:
            children = list(site.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.name.endswith(".dist-info") or not child.is_dir():
                continue
            base = child.name[: -len(".dist-info")]
            if not any(
                base.startswith(f"{pkg}-")
                or base.startswith(f"{pkg.replace('-', '_')}-")
                for pkg in package_names
            ):
                continue
            direct_url = child / "direct_url.json"
            try:
                if not direct_url.is_file():
                    continue
                data = json.loads(direct_url.read_text(encoding="utf-8"))
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
