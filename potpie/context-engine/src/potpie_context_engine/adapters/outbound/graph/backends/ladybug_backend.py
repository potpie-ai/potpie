"""Ladybug persistence with the common local graph mutation semantics."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.core.definition import GraphDefinition
from potpie_context_engine.core.graph_journal import encode_journal
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

from .._local_json_atomic import locked_json_store
from .._mutation_execution import execution_receipts_to_json
from ..ladybug_storage import ensure_schema, execute, read_payload, write_payload
from ..ladybug_windows_bootstrap import _require_ladybug
from .embedded_backend import EmbeddedGraphBackend
from .in_memory_backend import dump_store


@dataclass(slots=True)
class LadybugGraphBackend(EmbeddedGraphBackend):
    """Persistent native graph; exact shared semantic ranking and filtering.

    Each operation opens under the process lock and closes before releasing it.
    This permits independent CLI and daemon processes to share an embedded DB.
    Full graph hydration/replacement currently favors correctness over scale.
    """

    db_path: Path | None = None
    _connection: Any = field(default=None, init=False, repr=False)
    _transaction_open: bool = field(default=False, init=False, repr=False)

    @property
    def profile(self) -> str:
        return "ladybug"

    @property
    def _path(self) -> Path:
        return (
            Path(self.db_path) if self.db_path is not None else self.home / "ladybug.db"
        )

    @contextmanager
    def _locked_store(self):
        with locked_json_store(self._path):
            if self._state_observed and not self._path.exists():
                raise MutationPersistenceError(
                    "previously observed Ladybug graph state is missing"
                )
            native = _require_ladybug()
            db = native.Database(
                str(self._path),
                buffer_pool_size=128 * 1024 * 1024,
                max_num_threads=2,
                backend="pybind",
            )
            conn = None
            try:
                conn = native.Connection(db)
                self._connection = conn
                ensure_schema(conn)
                execute(conn, "BEGIN TRANSACTION")
                self._transaction_open = True
                yield
                # Read-only operations have nothing to publish. Mutation writes
                # commit in _write_state, before the base class adopts them.
            finally:
                try:
                    if conn is not None:
                        if self._transaction_open:
                            # Preserve an ambiguous commit acknowledgement;
                            # closing discards any remaining transaction.
                            with suppress(Exception):
                                execute(conn, "ROLLBACK")
                            self._transaction_open = False
                        self._connection = None
                        conn.close()
                finally:
                    db.close()

    def _load_state(self):
        try:
            state = self._decode_state(read_payload(self._connection))
        except MutationPersistenceError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise MutationPersistenceError("invalid Ladybug graph state") from exc
        self._state_observed = True
        return state

    def _write_state(self, store, registry, *, revisions=None, journal_data=None):
        payload = {
            "schema_version": 1,
            **dump_store(store),
            "mutation_receipts": execution_receipts_to_json(registry.completed()),
            "graph_revisions": dict(
                revisions if revisions is not None else self._revisions
            ),
            "graph_journal": encode_journal(
                journal_data if journal_data is not None else self._journal_data
            ),
        }
        try:
            write_payload(self._connection, payload)
            execute(self._connection, "COMMIT")
            self._transaction_open = False
        except Exception as exc:
            # A commit error may be an unknown acknowledgement. Leave live state
            # untouched; the next operation reloads authoritative native records.
            raise MutationPersistenceError("Ladybug graph transaction failed") from exc

    def bind_definition(self, definition: GraphDefinition) -> LadybugGraphBackend:
        return LadybugGraphBackend(
            home=self.home,
            db_path=self.db_path,
            embedder=self.embedder,
            definition=definition,
            shared_store=self._inner.store,
            shared_execution_registry=self._inner.execution_registry,
            resource_exists=self.resource_exists,
        )

    def provision(self) -> BackendProvisionResult:
        self._transact(lambda _inner: None)
        return BackendProvisionResult(
            ok=True,
            detail=f"Ladybug store at {self._path}",
            metadata={"profile": self.profile, "path": str(self._path)},
        )


__all__ = ["LadybugGraphBackend"]
