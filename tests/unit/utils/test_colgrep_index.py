"""Unit tests for ColGREP index helper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.modules.utils import colgrep_index


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_colgrep_xdg_data_home_under_repos_base(tmp_path: Path) -> None:
    repos = tmp_path / "my_repos"
    repos.mkdir()
    xdg = colgrep_index.colgrep_xdg_data_home(repos)
    assert xdg == repos / ".colgrep" / "xdg-data"
    assert xdg.is_dir()


def test_build_colgrep_index_skips_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLGREP_DISABLE_INDEX", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    with patch.object(colgrep_index, "subprocess") as mock_sp:
        colgrep_index.build_colgrep_index(str(repo), repos_base_path=tmp_path / "r")
        mock_sp.run.assert_not_called()


def test_build_colgrep_index_skips_without_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLGREP_DISABLE_INDEX", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    with patch.object(colgrep_index, "resolve_colgrep_binary", return_value=None):
        with patch.object(colgrep_index, "subprocess") as mock_sp:
            colgrep_index.build_colgrep_index(str(repo), repos_base_path=tmp_path / "r")
            mock_sp.run.assert_not_called()


def test_build_colgrep_index_runs_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLGREP_DISABLE_INDEX", raising=False)
    monkeypatch.setenv("COLGREP_INIT_TIMEOUT_SEC", "60")
    repos_base = tmp_path / "repos"
    repos_base.mkdir()
    repo = repos_base / "wt"
    repo.mkdir()
    completed = MagicMock()
    completed.returncode = 0
    completed.stderr = ""
    completed.stdout = ""
    with patch.object(colgrep_index, "resolve_colgrep_binary", return_value="/usr/bin/colgrep"):
        with patch.object(
            colgrep_index.subprocess,
            "run",
            return_value=completed,
        ) as mock_run:
            colgrep_index.build_colgrep_index(str(repo), repos_base_path=repos_base)
    mock_run.assert_called_once()
    _args, kwargs = mock_run.call_args
    assert kwargs["env"]["XDG_DATA_HOME"] == str(
        repos_base / ".colgrep" / "xdg-data"
    )
    assert list(_args[0]) == ["/usr/bin/colgrep", "init", "-y", str(repo.resolve())]


def test_resolve_colgrep_binary_prefers_repo_local_binary_over_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_binary = _make_executable(tmp_path / "colgrep")
    monkeypatch.delenv("COLGREP_BINARY", raising=False)
    monkeypatch.setattr(
        colgrep_index,
        "default_colgrep_binary_path",
        lambda: local_binary,
    )
    monkeypatch.setattr(colgrep_index.shutil, "which", lambda _: "/usr/bin/colgrep")

    resolved = colgrep_index.resolve_colgrep_binary()

    assert resolved == str(local_binary)


def test_resolve_sandbox_colgrep_binary_prefers_matching_packaged_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm64_binary = _make_executable(tmp_path / "colgrep-linux-arm64")
    monkeypatch.delenv("COLGREP_SANDBOX_BINARY", raising=False)
    monkeypatch.setattr(
        colgrep_index,
        "_packaged_linux_binary_candidates",
        lambda: [(arm64_binary, "linux/arm64/v8")],
    )

    resolved_binary, docker_platform = colgrep_index.resolve_sandbox_colgrep_binary()

    assert resolved_binary == str(arm64_binary)
    assert docker_platform == "linux/arm64/v8"
