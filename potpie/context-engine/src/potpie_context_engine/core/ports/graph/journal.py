"""Optional native journal and rebuildable header-mirror ports."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from potpie_context_engine.core.graph_journal import (
    CommitReceipt,
    JournalCapability,
    JournalLimits,
    JournalState,
)
from potpie_context_engine.core.graph_restore import RestorePlan


class GraphJournalPort(Protocol):
    def activate(
        self,
        *,
        pot_id: str,
        rollback_enabled: bool = False,
        limits: JournalLimits | None = None,
    ) -> JournalState: ...
    def journal_capability(self, pot_id: str) -> JournalCapability: ...
    def set_rollback_enabled(self, *, pot_id: str, enabled: bool) -> JournalState: ...
    def journal_state(self, pot_id: str) -> JournalState | None: ...
    def get_receipt(self, *, pot_id: str, commit_id: str) -> CommitReceipt | None: ...
    def read_receipts(
        self, *, pot_id: str, generation: str, after_sequence: int, limit: int = 100
    ) -> Sequence[CommitReceipt]: ...
    def plan_restore(
        self,
        *,
        pot_id: str,
        target_commit_id: str,
        expected_head: str,
        mode: str = "revert",
    ) -> RestorePlan: ...
    def apply_restore(
        self, plan: RestorePlan, *, mutation_id: str, actor: str
    ) -> CommitReceipt: ...
    def begin_resource(
        self, *, pot_id: str, operation_id: str, owner: str
    ) -> JournalState: ...
    def complete_resource(
        self, *, pot_id: str, operation_id: str, owner: str
    ) -> CommitReceipt: ...
    def recover_resource_guard(
        self,
        *,
        pot_id: str,
        operation_id: str,
        expected_owner: str,
        new_owner: str,
        verify_worker_stopped: Callable[[str], bool],
    ) -> JournalState: ...
