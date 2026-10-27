"""Fail clearly for mixed-architecture packages before touching a graph store."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph import falkordb_native
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    build_falkordb_graph,
)
from potpie_context_engine.core.errors import GraphSubstrateUnavailable


def _elf(path: Path, machine: int, *, byteorder: str = "little") -> str:
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02" + bytes([1 if byteorder == "little" else 2, 1])
    header[16:18] = (3).to_bytes(2, byteorder)
    header[18:20] = machine.to_bytes(2, byteorder)
    path.write_bytes(header)
    return str(path)


def _client(monkeypatch, executable: str, module: str) -> None:
    package = ModuleType("redislite")
    package.client = SimpleNamespace(
        __redis_executable__=executable, __falkordb_module__=module
    )
    monkeypatch.setitem(sys.modules, "redislite", package)
    monkeypatch.setattr(falkordb_native.platform, "system", lambda: "Linux")


def test_overlapping_distributions_fail_before_import_and_preserve_store(
    monkeypatch, tmp_path
) -> None:
    versions = {"falkordblite": "0.10.0", "potpie-falkordblite": "0.10.0.post1"}
    monkeypatch.setattr(
        falkordb_native.metadata,
        "distribution",
        lambda name: SimpleNamespace(version=versions[name]),
    )
    monkeypatch.setattr(falkordb_native.platform, "system", lambda: "Linux")
    # A partially uninstalled package must still receive the ownership error,
    # rather than importing its possibly overwritten or deleted driver files.
    monkeypatch.setitem(sys.modules, "redislite", None)
    monkeypatch.setitem(sys.modules, "redislite.falkordb_client", None)
    store = tmp_path / "graph.db"
    store.write_bytes(b"existing graph data")
    settings = SimpleNamespace(
        falkordb_mode=lambda: "lite",
        falkordb_graph_name=lambda: "graph",
        falkordb_lite_path=lambda: str(store),
    )

    with pytest.raises(GraphSubstrateUnavailable) as error:
        build_falkordb_graph(settings)

    assert "Both falkordblite==0.10.0 and potpie-falkordblite==0.10.0.post1" in str(
        error.value
    )
    action = error.value.recommended_next_action
    assert "python -m pip uninstall falkordblite" in action
    assert (
        "python -m pip install --force-reinstall --no-deps "
        "potpie-falkordblite==0.10.0.post1" in action
    )
    assert "same trusted index or --find-links wheel directory" in action
    assert store.read_bytes() == b"existing graph data"


@pytest.mark.parametrize("owner", ["falkordblite", "potpie-falkordblite"])
def test_single_distribution_installation_is_allowed(monkeypatch, owner):
    def distribution(name):
        if name == owner:
            return SimpleNamespace(version="0.10.0.post1")
        raise falkordb_native.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(falkordb_native.metadata, "distribution", distribution)
    monkeypatch.setattr(falkordb_native.platform, "system", lambda: "Linux")
    falkordb_native.validate_falkordb_lite_installation()


def test_mixed_package_fails_before_starting_or_creating_store(
    monkeypatch, tmp_path
) -> None:
    server = _elf(tmp_path / "redis-server", 183)
    module = _elf(tmp_path / "falkordb.so", 62)
    _client(monkeypatch, server, module)
    driver = ModuleType("redislite.falkordb_client")

    def unexpected_server(*args, **kwargs):
        raise AssertionError("incompatible native server must not start")

    driver.FalkorDB = unexpected_server
    monkeypatch.setitem(sys.modules, "redislite.falkordb_client", driver)
    store = tmp_path / "unopened" / "graph.db"
    settings = SimpleNamespace(
        falkordb_mode=lambda: "lite",
        falkordb_graph_name=lambda: "graph",
        falkordb_lite_path=lambda: str(store),
    )

    with pytest.raises(GraphSubstrateUnavailable) as error:
        build_falkordb_graph(settings)

    assert "ARM64" in str(error.value)
    assert "x86-64" in str(error.value)
    assert repr(server) in str(error.value)
    assert repr(module) in str(error.value)
    assert "corrected FalkorDBLite wheel" in error.value.recommended_next_action
    assert not store.parent.exists()
    assert Path(server).read_bytes()[18:20] == b"\xb7\x00"
    assert Path(module).read_bytes()[18:20] == b"\x3e\x00"


@pytest.mark.parametrize("machine", [62, 183])
def test_matching_binaries_allowed_including_emulated_pair(
    monkeypatch, tmp_path, machine
) -> None:
    _client(
        monkeypatch,
        _elf(tmp_path / "redis-server", machine),
        _elf(tmp_path / "falkordb.so", machine),
    )
    monkeypatch.setattr(falkordb_native.platform, "machine", lambda: "aarch64")
    falkordb_native.validate_falkordb_lite_binaries()


def test_same_machine_different_byte_order_is_incompatible(monkeypatch, tmp_path):
    _client(
        monkeypatch,
        _elf(tmp_path / "redis-server", 183),
        _elf(tmp_path / "falkordb.so", 183, byteorder="big"),
    )
    with pytest.raises(GraphSubstrateUnavailable, match="big-endian"):
        falkordb_native.validate_falkordb_lite_binaries()


def test_path_fallback_checks_the_executable_driver_will_use(monkeypatch, tmp_path):
    server = _elf(tmp_path / "redis-server", 183)
    _client(monkeypatch, "", _elf(tmp_path / "falkordb.so", 62))
    monkeypatch.setattr(falkordb_native.shutil, "which", lambda name: server)
    with pytest.raises(GraphSubstrateUnavailable, match="incompatible native"):
        falkordb_native.validate_falkordb_lite_binaries()


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_other_platforms_do_not_import_or_inspect_binaries(monkeypatch, system):
    monkeypatch.setitem(sys.modules, "redislite", None)
    monkeypatch.setattr(falkordb_native.platform, "system", lambda: system)
    monkeypatch.setattr(
        falkordb_native.metadata,
        "distribution",
        lambda name: pytest.fail("ownership inspection is Linux-only"),
    )
    falkordb_native.validate_falkordb_lite_installation()
    falkordb_native.validate_falkordb_lite_binaries()


@pytest.mark.parametrize("contents", [b"", b"\x7fELF", b"#!/bin/sh\n", b"MZ" * 32])
def test_non_elf_or_truncated_executable_keeps_driver_diagnostics(
    monkeypatch, tmp_path, contents
):
    server = tmp_path / "redis-server"
    server.write_bytes(contents)
    _client(monkeypatch, str(server), _elf(tmp_path / "falkordb.so", 183))
    falkordb_native.validate_falkordb_lite_binaries()


def test_missing_module_keeps_driver_diagnostics(monkeypatch, tmp_path):
    _client(
        monkeypatch,
        _elf(tmp_path / "redis-server", 183),
        str(tmp_path / "missing.so"),
    )
    falkordb_native.validate_falkordb_lite_binaries()


def test_server_mode_does_not_inspect_local_binaries(monkeypatch):
    package = ModuleType("falkordb")
    result = object()
    package.FalkorDB = SimpleNamespace(
        from_url=lambda url: SimpleNamespace(select_graph=lambda name: result)
    )
    monkeypatch.setitem(sys.modules, "falkordb", package)
    monkeypatch.setitem(sys.modules, "redislite", None)
    settings = SimpleNamespace(
        falkordb_mode=lambda: "server",
        falkordb_graph_name=lambda: "graph",
        falkordb_url=lambda: "redis://127.0.0.1:6380",
    )
    assert build_falkordb_graph(settings) is result
