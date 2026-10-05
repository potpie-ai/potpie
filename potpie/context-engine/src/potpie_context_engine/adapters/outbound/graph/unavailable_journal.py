"""Explicit capability response for backend profiles without native receipts."""

from potpie_context_core.graph_journal import JournalCapability, JournalError


class UnavailableJournal:
    def __init__(self, reason: str):
        self.reason = reason

    def journal_capability(self, pot_id: str) -> JournalCapability:
        return JournalCapability(detail=self.reason)

    def journal_state(self, pot_id: str):
        return None

    def activate(self, **kwargs):
        raise JournalError(self.reason)

    def set_rollback_enabled(self, **kwargs):
        raise JournalError(self.reason)

    def get_receipt(self, **kwargs):
        raise JournalError(self.reason)

    def read_receipts(self, **kwargs):
        raise JournalError(self.reason)

    def plan_restore(self, **kwargs):
        raise JournalError(self.reason)

    def apply_restore(self, plan, **kwargs):
        raise JournalError(self.reason)

    def begin_resource(self, **kwargs):
        raise JournalError(self.reason)

    def complete_resource(self, **kwargs):
        raise JournalError(self.reason)

    def recover_resource_guard(self, **kwargs):
        raise JournalError(self.reason)
