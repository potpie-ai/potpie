"""Keep native-wheel acceptance bounded and clean up its working directory."""

# ruff: noqa: S101 - pytest assertions are the test oracle.

from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts import test_arm_wheels

pytestmark = pytest.mark.source_contract


def test_suite_reports_timeout_with_selection_and_report(tmp_path, monkeypatch):
    report = tmp_path / "reliability.xml"
    selections = ["tests/integration/test_falkordb_reliability.py", "-k", "not server"]
    test_file = tmp_path / selections[0]
    test_file.parent.mkdir(parents=True)
    test_file.touch()

    def run(command, *, cwd, check, timeout):
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert command[-len(selections) :] == [str(test_file), *selections[1:]]
        assert cwd == tmp_path
        assert check
        assert 0 < timeout <= 10 * 60
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(test_arm_wheels.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="reliability.*timed out") as error:
        test_arm_wheels.run_suite(tmp_path, report, selections)

    assert str(report) in str(error.value)
    assert selections[0] in str(error.value)
    assert "backend shutdown" in str(error.value)
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
    assert not report.exists()


@pytest.mark.parametrize(
    ("testcase", "failure"),
    [
        ('<testcase name="accepted"/>', None),
        ('<testcase name="required"><skipped/></testcase>', "Required acceptance"),
        ("", "No acceptance tests ran"),
    ],
)
def test_suite_validates_completed_report(tmp_path, monkeypatch, testcase, failure):
    report = tmp_path / "resources.xml"
    test_file = tmp_path / "tests/unit/test_resources.py"
    test_file.parent.mkdir(parents=True)
    test_file.touch()

    def run(command, **kwargs):
        report.write_text(f"<testsuite>{testcase}</testsuite>")

    monkeypatch.setattr(test_arm_wheels.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            test_arm_wheels.run_suite(
                tmp_path, report, ["tests/unit/test_resources.py"]
            )
    else:
        test_arm_wheels.run_suite(tmp_path, report, ["tests/unit/test_resources.py"])


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
