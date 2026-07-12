"""Embedded ``GraphBackend`` — the OSS local default (JSON-persisted POC).

The intended ``pip install potpie`` default: a local, no-Docker, persistent store
so the CLI round-trips ``record`` → ``resolve`` across separate invocations
(each ``potpie`` call is a fresh process). This POC reuses the in-memory
capability adapters over a claim store that loads from / saves to a JSON file at
``<home>/graph.json``, persisting after every mutation.

It is a real, working backend — the POC found that an ephemeral ``in_memory``
store cannot back a process-per-call CLI, and ``embedded`` is the documented
answer. It is intentionally simple (JSON file, naive vectors); the real embedded
store is SQLite + a local vector index.

    TODO(stage-N): replace the JSON file with SQLite + a local vector index and
    real embeddings; keep this capability surface unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    dump_store,
    load_store,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.domain.provisioning import DONE, StepResult
from potpie_context_engine.domain.ports.claim_query import ClaimQueryPort
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.graph.analytics import GraphAnalyticsPort
from potpie_context_engine.domain.ports.graph.backend import BackendCapabilities
from potpie_context_engine.domain.ports.graph.inspection import GraphInspectionPort
from potpie_context_engine.domain.ports.graph.mutation import GraphMutationPort
from potpie_context_engine.domain.ports.graph.semantic import SemanticSearchPort
from potpie_context_engine.domain.ports.graph.snapshot import GraphSnapshotPort

_PROFILE = "embedded"


@dataclass(slots=True)
class EmbeddedGraphBackend:
    """Local JSON-persisted backend; delegates capabilities to the in-memory
    adapters and saves the store after each mutation."""

    home: Path = field(default_factory=default_home)
    embedder: EmbedderPort | None = None
    _inner: InMemoryGraphBackend = field(init=False)

    def __post_init__(self) -> None:
        store = self._load_store()
        store.embedder = self.embedder
        self._inner = InMemoryGraphBackend(
            store=store,
            profile_name=_PROFILE,
            on_change=self._save_store,
            embedder=self.embedder,
        )

    @property
    def match_mode(self) -> str:
        return self._inner.match_mode

    @property
    def _path(self) -> Path:
        return self.home / "graph.json"

    def _load_store(self) -> InMemoryClaimQueryStore:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return load_store(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError):
            return InMemoryClaimQueryStore()

    def _save_store(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dump_store(self._inner.store), fh)
        tmp.replace(self._path)

    # --- capability bundle (delegate to the in-memory adapters) -------------
    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._inner.claim_query

    @property
    def mutation(self) -> GraphMutationPort:
        return self._inner.mutation

    @property
    def semantic(self) -> SemanticSearchPort:
        return self._inner.semantic

    @property
    def inspection(self) -> GraphInspectionPort:
        return self._inner.inspection

    @property
    def analytics(self) -> GraphAnalyticsPort:
        return self._inner.analytics

    @property
    def snapshot(self) -> GraphSnapshotPort:
        return self._inner.snapshot

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            semantic=True,
            inspection=True,
            analytics=True,
            snapshot=True,
        )

    def provision(self) -> StepResult:
        # Stand up the local store: ensure the home dir + persist the (possibly
        # empty) store file so resolve works immediately after setup. Idempotent.
        self.home.mkdir(parents=True, exist_ok=True)
        self._save_store()
        return StepResult(
            step="backend.provision",
            state=DONE,
            detail=f"embedded store at {self._path}",
            metadata={"profile": _PROFILE, "path": str(self._path)},
        )


__all__ = ["EmbeddedGraphBackend"]
