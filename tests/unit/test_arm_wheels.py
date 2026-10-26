"""The native-wheel runner must release its working directory before cleanup."""

from pathlib import Path
import os
import sys

import pytest

from scripts import test_arm_wheels


@pytest.mark.parametrize("suite_fails", [False, True])
def test_runner_restores_cwd_and_removes_scratch_directory(
    tmp_path, monkeypatch, suite_fails
):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "potpie/context-engine/tests").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_arm_wheels.py",
            "--repo",
            str(repo),
            "--target",
            "windows",
            "--reports",
            str(tmp_path / "reports"),
        ],
    )
    # Keep the runner's environment isolation local to this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    temporary_directory = test_arm_wheels.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        test_arm_wheels.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: temporary_directory(dir=tmp_path, **kwargs),
    )
    original_cwd = Path.cwd()
    scratch = []

    def verify_installation(target):
        assert target == "windows"
        scratch.append(Path.cwd())
        assert scratch[0] != original_cwd

    monkeypatch.setattr(test_arm_wheels, "verify_installation", verify_installation)
    suites = []

    def run_suite(directory, report, selections):
        assert Path.cwd() == scratch[0]
        assert directory.is_relative_to(scratch[0])
        assert (directory / "tests").is_dir()
        suites.append(selections)
        if suite_fails:
            raise RuntimeError("acceptance failed")

    monkeypatch.setattr(test_arm_wheels, "run_suite", run_suite)
    if suite_fails:
        with pytest.raises(RuntimeError, match="acceptance failed"):
            test_arm_wheels.main()
    else:
        test_arm_wheels.main()

    assert len(suites) == (1 if suite_fails else 3)
    assert Path.cwd() == original_cwd
    assert not scratch[0].exists()
