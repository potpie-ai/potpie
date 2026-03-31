from unittest.mock import MagicMock, patch

from app.modules.utils import gvisor_runner


def test_filter_safe_environment_variables_keeps_xdg_data_home(tmp_path):
    xdg_data_home = tmp_path / "xdg-data"
    filtered = gvisor_runner._filter_safe_environment_variables(
        {
            "PATH": "/usr/bin:/bin",
            "XDG_DATA_HOME": str(xdg_data_home),
        }
    )

    assert filtered["XDG_DATA_HOME"] == str(xdg_data_home)


def test_run_with_docker_gvisor_mounts_xdg_data_home(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    xdg_data_home = tmp_path / "xdg-data"
    xdg_data_home.mkdir()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = "ok"
    completed.stderr = ""

    with patch.object(gvisor_runner, "resolve_sandbox_colgrep_binary", return_value=(None, None)):
        with patch.object(gvisor_runner.shutil, "which", return_value=None):
            with patch.object(
                gvisor_runner.subprocess,
                "run",
                return_value=completed,
            ) as mock_run:
                result = gvisor_runner._run_with_docker_gvisor(
                    command=["colgrep", "search", "needle"],
                    working_dir=str(worktree),
                    repo_path=None,
                    env={
                        "PATH": "/usr/bin:/bin",
                        "XDG_DATA_HOME": str(xdg_data_home),
                    },
                    timeout=30,
                    runsc_path=None,
                )

    assert result.success is True
    docker_cmd = mock_run.call_args.args[0]
    assert (
        f"{xdg_data_home}:{gvisor_runner._DOCKER_XDG_DATA_HOME_MOUNT}:ro"
        in docker_cmd
    )
    assert f"XDG_DATA_HOME={gvisor_runner._DOCKER_XDG_DATA_HOME_MOUNT}" in docker_cmd
    assert f"XDG_DATA_HOME={xdg_data_home}" not in docker_cmd


def test_run_with_docker_gvisor_mounts_resolved_colgrep_binary(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    packaged_colgrep = tmp_path / "colgrep-linux-arm64"
    packaged_colgrep.write_text("binary")

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = "ok"
    completed.stderr = ""

    with patch.object(
        gvisor_runner,
        "resolve_sandbox_colgrep_binary",
        return_value=(str(packaged_colgrep), "linux/arm64/v8"),
    ):
        with patch.object(gvisor_runner.shutil, "which", return_value=None):
            with patch.object(
                gvisor_runner.subprocess,
                "run",
                return_value=completed,
            ) as mock_run:
                result = gvisor_runner._run_with_docker_gvisor(
                    command=["colgrep", "search", "needle"],
                    working_dir=str(worktree),
                    repo_path=None,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=30,
                    runsc_path=None,
                )

    assert result.success is True
    docker_cmd = mock_run.call_args.args[0]
    assert "--platform" in docker_cmd
    assert "linux/arm64/v8" in docker_cmd
    assert (
        f"{packaged_colgrep}:{gvisor_runner._DOCKER_COLGREP_BINARY_MOUNT}:ro"
        in docker_cmd
    )
    assert gvisor_runner._DOCKER_RUNTIME_IMAGE in docker_cmd
