"""Per-session injection ledger port (Graph V1.5 Step 12a / noise control).

A nudge that injects the same preference or bug twice in one session gets
ignored, so the brain records what it has injected per ``session_id`` and skips
repeats. This port is that memory; implementations persist it (local JSON file)
or hold it in memory (tests).
"""

from __future__ import annotations

from typing import Protocol, Sequence


class InjectionLedgerPort(Protocol):
    """Tracks which candidate keys have been injected within a session."""

    def was_injected(self, session_id: str, key: str) -> bool:
        """True iff ``key`` was already injected for ``session_id``."""
        ...

    def record(self, session_id: str, keys: Sequence[str]) -> None:
        """Mark ``keys`` as injected for ``session_id`` (idempotent)."""
        ...


__all__ = ["InjectionLedgerPort"]
