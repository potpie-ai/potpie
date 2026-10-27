"""Exercise CI selection and skip gates through an isolated pytest process."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.source_contract


def _run(tmp_path: Path, files: dict[str, str], tier: str | None):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").touch()
    shutil.copyfile(
        Path(__file__).parents[2] / "required_acceptance.py",
        tests / "required_acceptance.py",
    )
    for name, source in files.items():
        destination = tests / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(source)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-v",
        "-p",
        "tests.required_acceptance",
    ]
    if tier is not None:
        command.extend(["--acceptance-tier", tier])
    return subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("tier", "selected"),
    [
        (
            "general",
            {"test_general", "test_embedded_restart_and_failure_before_publication"},
        ),
        ("graph", {"test_graph"}),
        ("restart", {"test_serial"}),
    ],
)
def test_tiers_partition_tests_and_keep_unlisted_restart_tests(
    tmp_path, tier, selected
):
    result = _run(
        tmp_path,
        {
            "test_general.py": "def test_general(): pass\n",
            "test_optional_dependency.py": (
                "import pytest\n"
                "pytest.skip('optional dependency unavailable', allow_module_level=True)\n"
            ),
            "conformance/test_graph_journal.py": (
                "def test_graph(): pass\n"
                "def test_embedded_restart_and_failure_before_publication(): pass\n"
            ),
            "conformance/test_journal_crash.py": "def test_serial(): pass\n",
        },
        tier,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    executed = {
        line.split("::")[-1].split()[0]
        for line in result.stdout.splitlines()
        if " PASSED" in line
    }
    assert executed == selected


@pytest.mark.parametrize(
    ("tier", "file", "exit_code"),
    [
        ("general", "test_optional.py", 0),
        ("graph", "conformance/test_graph_reliability.py", 1),
        ("restart", "conformance/test_journal_crash.py", 1),
        (None, "test_windows_selection.py", 1),
    ],
)
def test_only_general_tier_allows_skips(tmp_path, tier, file, exit_code):
    result = _run(
        tmp_path,
        {file: "import pytest\ndef test_scenario(): pytest.skip('unavailable')\n"},
        tier,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert ("Required acceptance scenarios skipped" in result.stdout) == bool(exit_code)
