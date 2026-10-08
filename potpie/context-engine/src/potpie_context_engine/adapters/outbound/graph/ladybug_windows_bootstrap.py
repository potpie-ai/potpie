"""Windows-only OpenSSL bootstrap so Ladybug's native ``_lbug*.pyd`` can load.

Ladybug 0.19+ Windows wheels exclude OpenSSL DLLs that the extension still
imports. Register CPython's OpenSSL 3 DLLs via ``os.add_dll_directory()``
**before** ``import ladybug``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile

_REQUIRED_DLLS = {
    "libssl-3-x64.dll": "libssl-3.dll",
    "libcrypto-3-x64.dll": "libcrypto-3.dll",
    "libcrypto-3.dll": "libcrypto-3.dll",
}


def ensure_ladybug_openssl() -> dict:
    """Register OpenSSL DLLs for ladybug on Windows. No-op elsewhere."""
    if sys.platform != "win32":
        return {"ok": True, "skipped": "not_windows"}

    if _ladybug_vendors_openssl():
        return {"ok": True, "skipped": "ladybug_vendors_openssl"}

    # Detached daemon children receive POTPIE_LADYBUG_OPENSSL_DIR from the
    # parent launch env (add_dll_directory is not inherited across processes).
    preset = (os.environ.get("POTPIE_LADYBUG_OPENSSL_DIR") or "").strip()
    if preset and os.path.isdir(preset):
        os.add_dll_directory(preset)
        os.environ["PATH"] = preset + os.pathsep + os.environ.get("PATH", "")
        return {"ok": True, "cache_directory": preset, "source": "env"}

    source_directory = os.path.join(sys.base_prefix, "DLLs")
    candidate_dirs = [
        source_directory,
        os.path.join(sys.base_prefix, "Library", "bin"),
        os.path.join(sys.exec_prefix, "DLLs"),
    ]
    exe_dir = os.path.dirname(
        getattr(sys, "_base_executable", sys.executable) or sys.executable
    )
    if exe_dir:
        candidate_dirs.append(exe_dir)

    sources: dict[str, str] = {}
    for target, source_name in _REQUIRED_DLLS.items():
        found = None
        for directory in candidate_dirs:
            path = os.path.join(directory, source_name)
            if os.path.isfile(path):
                found = path
                break
        if found is None:
            sources = {}
            break
        sources[target] = found

    if not sources:
        return {
            "ok": False,
            "error": (
                "CPython OpenSSL DLLs (libssl-3.dll / libcrypto-3.dll) not found. "
                "Install python.org CPython 3.12 or 3.13 x64 (not Microsoft Store "
                "Python), then reinstall Potpie."
            ),
            "searched": candidate_dirs,
        }

    cache_directory = _cache_directory()
    os.makedirs(cache_directory, exist_ok=True)
    for target, source in sources.items():
        _copy_once(source, os.path.join(cache_directory, target))

    os.add_dll_directory(cache_directory)
    os.environ["PATH"] = cache_directory + os.pathsep + os.environ.get("PATH", "")
    os.environ["POTPIE_LADYBUG_OPENSSL_DIR"] = cache_directory
    return {"ok": True, "cache_directory": cache_directory}


def verify_ladybug_native() -> dict:
    """Ensure pybind backend loads (not the broken C-API fallback)."""
    status = ensure_ladybug_openssl()
    if not status.get("ok"):
        return status
    try:
        import ladybug  # noqa: F401
        from ladybug._backend import get_pybind_module

        mod = get_pybind_module()
        if mod is None:
            return {
                "ok": False,
                "error": (
                    "ladybug._lbug pybind extension failed to import even after "
                    "OpenSSL bootstrap; install VC++ Redistributable x64 and "
                    "confirm a cp312/cp313 win_amd64 wheel matches this Python."
                ),
                "openssl": status,
            }
        return {
            "ok": True,
            "ladybug": getattr(ladybug, "__version__", "?"),
            "backend": "pybind",
            "openssl": status,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "openssl": status}


def _cache_directory() -> str:
    tag = hashlib.sha256(sys.base_prefix.encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"potpie-ladybug-openssl-{tag}")


def _copy_once(source: str, target: str) -> None:
    if os.path.exists(target):
        return
    handle, staged = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".part")
    os.close(handle)
    try:
        shutil.copyfile(source, staged)
        os.replace(staged, target)
    except OSError:
        if not os.path.exists(target):
            raise
    finally:
        if os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass


def _ladybug_vendors_openssl() -> bool:
    spec = importlib.util.find_spec("ladybug")
    if spec is None or not spec.origin:
        return False
    package_directory = os.path.dirname(spec.origin)
    libs_directory = os.path.join(os.path.dirname(package_directory), "ladybug.libs")
    try:
        vendored = os.listdir(libs_directory)
    except OSError:
        return False
    return any(name.startswith("libssl-3-x64") for name in vendored)


__all__ = ["ensure_ladybug_openssl", "verify_ladybug_native"]
