from __future__ import annotations

import multiprocessing
from pathlib import Path

from potpie_context_engine.adapters.outbound.ledger.cursor_store import (
    LocalLedgerCursorStore,
)
from potpie_context_engine.domain.ports.ledger.client import LedgerCursor


def _set_cursor(home: str, index: int) -> None:
    LocalLedgerCursorStore(home=Path(home)).set(
        pot_id=f"pot-{index}",
        cursor=LedgerCursor(source_id=f"source-{index}", token=f"token-{index}"),
    )


def test_cursor_updates_are_serialized_across_processes(tmp_path: Path) -> None:
    process_context = multiprocessing.get_context("spawn")
    processes = [
        process_context.Process(target=_set_cursor, args=(str(tmp_path), index))
        for index in range(12)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    store = LocalLedgerCursorStore(home=tmp_path)
    for index in range(12):
        assert store.get(
            pot_id=f"pot-{index}", source_id=f"source-{index}"
        ) == LedgerCursor(source_id=f"source-{index}", token=f"token-{index}")
