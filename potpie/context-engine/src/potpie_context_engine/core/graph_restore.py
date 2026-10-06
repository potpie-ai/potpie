"""Trusted server-only restore types. Never load these from an RPC payload."""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_engine.core.graph_journal import JournalRecord


@dataclass(frozen=True, slots=True)
class RestoreRecord:
    before: JournalRecord
    after: JournalRecord


@dataclass(frozen=True, slots=True)
class RestorePlan:
    pot_id: str
    journal_generation: str
    resource_generation: int
    expected_head: str
    records: tuple[RestoreRecord, ...]
    target_commit_ids: tuple[str, ...]
    required_access: str
    inverse_hash: str
    mode: str
    target_commit_id: str
