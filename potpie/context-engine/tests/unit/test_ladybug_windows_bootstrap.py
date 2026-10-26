"""Windows native loading without altering the host's DLL search path."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph import (
    ladybug_windows_bootstrap as native,
)


@pytest.fixture
def windows_python(tmp_path, monkeypatch):
    python = tmp_path / "python"
    dlls = python / "DLLs"
    dlls.mkdir(parents=True)
    (dlls / "libssl-3.dll").write_bytes(b"ssl-version-one")
    (dlls / "libcrypto-3.dll").write_bytes(b"crypto-version-one")
    handles = {}
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.sys, "base_prefix", str(python))
    monkeypatch.setattr(native.sys, "exec_prefix", str(python))
    monkeypatch.setattr(native.sys, "_base_executable", str(python / "python.exe"))
    monkeypatch.setattr(native, "default_home", lambda: tmp_path / "home")
    monkeypatch.setattr(native, "_DLL_HANDLES", {})

    def register(path):
        assert path not in handles, "directory should be registered only once"
        handles[path] = object()
        return handles[path]

    monkeypatch.setattr(native.os, "add_dll_directory", register, raising=False)
    return dlls, handles


def test_openssl_aliases_verified_and_directory_handle_retained(windows_python):
    dlls, handles = windows_python
    native.ensure_ladybug_openssl()
    directory = Path(next(iter(handles)))
    assert (directory / "libssl-3-x64.dll").read_bytes() == (
        dlls / "libssl-3.dll"
    ).read_bytes()
    assert (directory / "libcrypto-3-x64.dll").read_bytes() == (
        dlls / "libcrypto-3.dll"
    ).read_bytes()
    assert native._DLL_HANDLES == handles

    # Repeated loads reuse the retained handle but verify existing file bytes.
    (directory / "libcrypto-3-x64.dll").write_bytes(b"stale-cache")
    native.ensure_ladybug_openssl()
    assert (directory / "libcrypto-3-x64.dll").read_bytes() == b"crypto-version-one"
    assert len(handles) == 1

    # A Python OpenSSL upgrade gets a new immutable directory.
    (dlls / "libssl-3.dll").write_bytes(b"ssl-version-two")
    native.ensure_ladybug_openssl()
    assert len(handles) == 2


def test_missing_cpython_dlls_fail_with_install_action(windows_python):
    dlls, handles = windows_python
    (dlls / "libssl-3.dll").unlink()
    with pytest.raises(ImportError, match="python.org"):
        native.ensure_ladybug_openssl()
    assert not handles


def test_non_windows_does_not_touch_dlls(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "linux")
    monkeypatch.setattr(
        native, "_openssl_sources", lambda: pytest.fail("unexpected DLL lookup")
    )
    native.ensure_ladybug_openssl()


@pytest.mark.parametrize("winner", [b"expected DLL", b"different DLL", None])
def test_concurrent_dll_publication_accepts_only_identical_loaded_winner(
    tmp_path, monkeypatch, winner
):
    target = tmp_path / "alias.dll"

    def publish_then_lock(source, destination):
        # A second process wins publication and loads the DLL between the
        # existence check and replacement, which locks the file on Windows.
        if winner is not None:
            destination.write_bytes(winner)
        raise PermissionError("DLL loaded by another process")

    monkeypatch.setattr(native.os, "replace", publish_then_lock)
    if winner == b"expected DLL":
        native._copy_verified(b"expected DLL", target)
        assert target.read_bytes() == winner
    else:
        with pytest.raises(PermissionError, match="DLL loaded"):
            native._copy_verified(b"expected DLL", target)
    assert not list(tmp_path.glob("*.tmp"))


def test_working_native_binding_skips_bootstrap(monkeypatch):
    package = SimpleNamespace()
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.importlib, "import_module", lambda name: package)
    monkeypatch.setattr(
        native, "ensure_ladybug_openssl", lambda: pytest.fail("unexpected bootstrap")
    )
    assert native._require_ladybug() is package


def test_failed_native_import_retries_after_bootstrap(monkeypatch):
    calls = []
    package = SimpleNamespace()

    def load(name):
        calls.append(name)
        if name == "ladybug._lbug" and "bootstrap" not in calls:
            raise ImportError("DLL missing")
        return package

    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.importlib, "import_module", load)
    monkeypatch.setattr(
        native, "ensure_ladybug_openssl", lambda: calls.append("bootstrap")
    )
    assert native._require_ladybug() is package
    assert calls == ["ladybug", "ladybug._lbug", "bootstrap", "ladybug._lbug"]


def test_missing_package_is_actionable(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(native.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"potpie-context-engine\[ladybug\]"):
        native._require_ladybug()
