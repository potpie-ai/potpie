"""Load Ladybug's native binding, supplying CPython's OpenSSL on Windows.

Ladybug 0.20 Windows wheels exclude OpenSSL. CPython supplies compatible
OpenSSL 3 libraries under different filenames. Keep any DLL search-directory
handles alive for the process; closing them removes the directory again.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.local_paths import default_home

_LOAD_LOCK = threading.RLock()
_DLL_HANDLES: dict[str, Any] = {}
_ALIASES = {
    "libssl-3-x64.dll": "libssl-3.dll",
    "libcrypto-3-x64.dll": "libcrypto-3.dll",
    "libcrypto-3.dll": "libcrypto-3.dll",
}


def _openssl_sources() -> dict[str, Path]:
    executable = getattr(sys, "_base_executable", None) or sys.executable
    directories = (
        Path(sys.base_prefix) / "DLLs",
        Path(sys.base_prefix) / "Library" / "bin",
        Path(sys.exec_prefix) / "DLLs",
        Path(executable).parent,
    )
    sources = {}
    for target, source_name in _ALIASES.items():
        candidates = (
            directory / name
            for directory in directories
            for name in dict.fromkeys((source_name, target))
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise ImportError(
                "Ladybug needs OpenSSL 3 DLLs on Windows. Install the x64 "
                "python.org Python distribution and the Microsoft Visual C++ "
                "Redistributable, then reinstall potpie. "
                f"Could not find {source_name} in the Python installation."
            )
        sources[target] = source
    return sources


def _copy_verified(content: bytes, target: Path) -> None:
    """Atomically replace absent or stale aliases; never trust an old cache."""
    if target.is_file() and target.read_bytes() == content:
        return
    descriptor, name = tempfile.mkstemp(dir=target.parent, suffix=".dll.tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        try:
            os.replace(temporary, target)
        except OSError:
            # Another process may publish and load the same DLL after our
            # initial check. Windows then prevents replacing the loaded file.
            # Accept only a byte-identical winner; stale or unreadable aliases
            # must still fail instead of being used as native dependencies.
            if not target.is_file() or target.read_bytes() != content:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def ensure_ladybug_openssl() -> None:
    """Register content-addressed CPython DLL aliases on Windows only."""
    if sys.platform != "win32":
        return
    with _LOAD_LOCK:
        contents = {
            name: path.read_bytes() for name, path in _openssl_sources().items()
        }
        digest = hashlib.sha256()
        for name, content in sorted(contents.items()):
            digest.update(name.encode("ascii"))
            digest.update(content)
        directory = default_home() / "native" / "ladybug-openssl" / digest.hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            _copy_verified(content, directory / name)
        key = str(directory)
        if key not in _DLL_HANDLES:
            _DLL_HANDLES[key] = os.add_dll_directory(key)


def _require_ladybug() -> Any:
    """Require the tested native binding instead of silently using C-API fallback."""
    with _LOAD_LOCK:
        try:
            module = importlib.import_module("ladybug")
        except ImportError as exc:
            raise ImportError(
                "Ladybug requires its optional native package. Install "
                "'potpie-context-engine[ladybug]' (included by potpie on Windows)."
            ) from exc
        try:
            importlib.import_module("ladybug._lbug")
        except ImportError:
            if sys.platform != "win32":
                raise
            ensure_ladybug_openssl()
            try:
                importlib.import_module("ladybug._lbug")
            except ImportError as exc:
                raise ImportError(
                    "Ladybug's native Windows binding could not load. Install "
                    "x64 Python and the Microsoft Visual C++ Redistributable; "
                    "ensure the Ladybug wheel matches your Python version."
                ) from exc
        return module
