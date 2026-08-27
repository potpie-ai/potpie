from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import json
import multiprocessing
from pathlib import Path

from potpie.config.local import LocalConfigService
from potpie.pots.local_store import LocalPotStore


def _set_config(home: str, key: str, value: str) -> None:
    LocalConfigService(home=Path(home)).set(key, value)


def _create_pot(home: str, name: str) -> None:
    LocalPotStore(home=Path(home)).create(name=name)


def _run_processes(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0


def test_config_updates_are_serialized_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_set_config,
            args=(str(tmp_path), f"key-{index}", f"value-{index}"),
        )
        for index in range(6)
    ]

    _run_processes(processes)

    state = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert state == {f"key-{index}": f"value-{index}" for index in range(6)}
    assert list(tmp_path.glob(".config.json.*.tmp")) == []


def test_pot_updates_are_serialized_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_create_pot,
            args=(str(tmp_path), f"pot-{index}"),
        )
        for index in range(6)
    ]

    _run_processes(processes)

    pots = LocalPotStore(home=tmp_path).list_pots()
    assert {pot["name"] for pot in pots} == {f"pot-{index}" for index in range(6)}
    assert len(pots) == 6
    assert list(tmp_path.glob(".pots.json.*.tmp")) == []
