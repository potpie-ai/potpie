"""``potpie doctor``'s PATH check, which has to work where the executable is
``potpie.exe`` and the shell has no ``which -a``."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from potpie.cli import cli_install_status as cis


def _executable(directory: Path, name: str = "potpie") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_every_potpie_on_path_is_listed_in_path_order(tmp_path: Path) -> None:
    first = _executable(tmp_path / "a")
    second = _executable(tmp_path / "b")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "potpie").write_text("not executable", encoding="utf-8")
    path_env = os.pathsep.join([str(tmp_path / "a"), "", str(tmp_path / "c"), str(tmp_path / "b")])

    found = cis._potpie_paths_on_path(path_env)

    assert [Path(p) for p in found] == [first, second]


def test_the_same_binary_reached_twice_is_listed_once(tmp_path: Path) -> None:
    real = _executable(tmp_path / "real")
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    (alias_dir / "potpie").symlink_to(real)

    found = cis._potpie_paths_on_path(os.pathsep.join([str(alias_dir), str(tmp_path / "real")]))

    assert len(found) == 1


def test_lookup_is_resolved_per_directory_with_which(monkeypatch, tmp_path: Path) -> None:
    """Resolution goes through ``shutil.which`` per directory -- which is what
    appends ``.exe``/``.cmd`` from PATHEXT on Windows. A bare join of the name
    found nothing there, so ``doctor`` said NOT on PATH on every Windows install."""
    asked: list[tuple[str, str | None]] = []

    def _which(name: str, path: str | None = None) -> str | None:
        asked.append((name, path))
        return str(Path(path or "") / "potpie.exe") if path and path.endswith("bin") else None

    monkeypatch.setattr(cis.shutil, "which", _which)
    found = cis._potpie_paths_on_path(os.pathsep.join([str(tmp_path / "nope"), str(tmp_path / "bin")]))

    assert [name for name, _ in asked] == ["potpie", "potpie"]
    assert found == [str(tmp_path / "bin" / "potpie.exe")]


def test_diagnostic_commands_follow_the_shell(monkeypatch) -> None:
    monkeypatch.setattr(cis.os, "name", "nt")
    windows = cis._diagnostic_commands()
    assert "where.exe potpie" in windows
    assert not any("$(" in cmd or "which -a" in cmd for cmd in windows)

    monkeypatch.setattr(cis.os, "name", "posix")
    posix = cis._diagnostic_commands()
    assert "which -a potpie" in posix
