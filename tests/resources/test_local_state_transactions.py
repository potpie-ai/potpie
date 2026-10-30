from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import json
import subprocess
import sys
from pathlib import Path

from potpie.pots.local_store import LocalPotStore


# Fresh interpreters import the installed product, never the pytest test package.
# This keeps the real cross-process lock contract usable under isolated wheel runs.
_WORKER = """
import sys
from pathlib import Path
from potpie.config.local import LocalConfigService
from potpie.pots.local_store import LocalPotStore
mode, home, key, *values = sys.argv[1:]
if mode == 'config':
    LocalConfigService(home=Path(home)).set(key, values[0])
else:
    LocalPotStore(home=Path(home)).create(name=key)
"""


def _run_processes(arguments: list[tuple[str, ...]]) -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", _WORKER, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for args in arguments
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_config_updates_are_serialized_across_processes(tmp_path: Path) -> None:
    _run_processes(
        [
            ("config", str(tmp_path), f"key-{index}", f"value-{index}")
            for index in range(6)
        ]
    )

    state = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert state == {f"key-{index}": f"value-{index}" for index in range(6)}
    assert list(tmp_path.glob(".config.json.*.tmp")) == []


def test_pot_updates_are_serialized_across_processes(tmp_path: Path) -> None:
    _run_processes([("pot", str(tmp_path), f"pot-{index}") for index in range(6)])

    pots = LocalPotStore(home=tmp_path).list_pots()
    assert {pot["name"] for pot in pots} == {f"pot-{index}" for index in range(6)}
    assert len(pots) == 6
    assert list(tmp_path.glob(".pots.json.*.tmp")) == []
