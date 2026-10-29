"""``potpie doctor``'s PATH check works where the executable is ``potpie.exe``."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from potpie.cli import cli_install_status as cis

_WINDOWS = sys.platform == "win32"


def _executable(directory: Path) -> Path:
    """A ``potpie`` the host's own ``which`` resolves: ``.exe`` on Windows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("potpie.exe" if _WINDOWS else "potpie")
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_every_potpie_on_path_is_listed_in_path_order(tmp_path: Path) -> None:
    first = _executable(tmp_path / "a")
    second = _executable(tmp_path / "b")
    (tmp_path / "empty").mkdir()
    path_env = os.pathsep.join(
        [str(tmp_path / "a"), "", str(tmp_path / "empty"), str(tmp_path / "b")]
    )

    found = cis._potpie_paths_on_path(path_env)

    assert [Path(p).resolve() for p in found] == [first.resolve(), second.resolve()]


@pytest.mark.skipif(_WINDOWS, reason="the execute bit is a POSIX concept")
def test_a_non_executable_potpie_is_not_on_path(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "potpie").write_text("not executable", encoding="utf-8")

    assert cis._potpie_paths_on_path(str(tmp_path / "plain")) == []


@pytest.mark.skipif(_WINDOWS, reason="symlinks need extra privileges on Windows")
def test_the_same_binary_reached_twice_is_listed_once(tmp_path: Path) -> None:
    real = _executable(tmp_path / "real")
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    (alias_dir / "potpie").symlink_to(real)

    found = cis._potpie_paths_on_path(
        os.pathsep.join([str(alias_dir), str(tmp_path / "real")])
    )

    assert len(found) == 1


def test_lookup_is_resolved_per_directory_with_which(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``shutil.which`` appends ``.exe``/``.cmd`` from PATHEXT on Windows; a
    bare join of the name found nothing there and reported NOT on PATH."""
    asked: list[tuple[str, str | None]] = []
    bin_dir = tmp_path / "bin"

    def _which(name: str, path: str | None = None) -> str | None:
        asked.append((name, path))
        return str(Path(path) / "potpie.exe") if path == str(bin_dir) else None

    monkeypatch.setattr(cis.shutil, "which", _which)
    found = cis._potpie_paths_on_path(
        os.pathsep.join([str(tmp_path / "nope"), str(bin_dir)])
    )

    assert asked == [("potpie", str(tmp_path / "nope")), ("potpie", str(bin_dir))]
    assert found == [str(bin_dir / "potpie.exe")]


def test_diagnostic_commands_follow_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scoped tightly: a patched ``os.name`` also changes how ``pathlib`` builds
    # paths, so nothing else may run while it is in place.
    with monkeypatch.context() as patch:
        patch.setattr(cis.os, "name", "nt")
        windows = cis._diagnostic_commands()
    with monkeypatch.context() as patch:
        patch.setattr(cis.os, "name", "posix")
        posix = cis._diagnostic_commands()

    assert "where.exe potpie" in windows
    assert not any("$(" in cmd or "which -a" in cmd for cmd in windows)
    assert "which -a potpie" in posix
