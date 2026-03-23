"""Unit tests for ColGREP index helper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.modules.utils import colgrep_index


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
    with patch.object(colgrep_index.shutil, "which", return_value=None):
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
    with patch.object(colgrep_index.shutil, "which", return_value="/usr/bin/colgrep"):
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
    assert list(_args[0]) == ["colgrep", "init", "-y", str(repo.resolve())]
