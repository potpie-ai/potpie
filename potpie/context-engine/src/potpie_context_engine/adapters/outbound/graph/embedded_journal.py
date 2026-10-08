"""Journal port backed by the embedded backend's existing file transaction."""

from potpie_context_engine.adapters.outbound.graph._local_json_atomic import (
    locked_json_store,
)


class EmbeddedJournal:
    def __init__(self, backend):
        self.backend = backend

    def _read(self, name, **kwargs):
        with locked_json_store(self.backend._path):
            store, registry = self.backend._load_state()
            inner = self.backend._transaction_backend(store, registry)
            result = getattr(inner.journal, name)(**kwargs)
            self.backend._adopt_state(store, registry)
            return result

    def _write(self, name, **kwargs):
        return self.backend._transact(
            lambda inner: getattr(inner.journal, name)(**kwargs)
        )

    def journal_capability(self, pot_id):
        return self._read("journal_capability", pot_id=pot_id)

    def journal_state(self, pot_id):
        return self._read("journal_state", pot_id=pot_id)

    def get_receipt(self, **kwargs):
        return self._read("get_receipt", **kwargs)

    def read_receipts(self, **kwargs):
        return self._read("read_receipts", **kwargs)

    def plan_restore(self, **kwargs):
        return self._read("plan_restore", **kwargs)

    def activate(self, **kwargs):
        return self._write("activate", **kwargs)

    def set_rollback_enabled(self, **kwargs):
        return self._write("set_rollback_enabled", **kwargs)

    def apply_restore(self, plan, **kwargs):
        return self.backend._transact(
            lambda inner: inner.journal.apply_restore(plan, **kwargs)
        )

    def begin_resource(self, **kwargs):
        return self._write("begin_resource", **kwargs)

    def complete_resource(self, **kwargs):
        return self._write("complete_resource", **kwargs)

    def recover_resource_guard(self, **kwargs):
        return self._write("recover_resource_guard", **kwargs)
