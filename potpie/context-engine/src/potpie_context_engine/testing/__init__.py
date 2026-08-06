"""Public backend authoring kit and reusable in-memory runtime fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from typing import Callable

from potpie_context_core.api import (
    Chunk,
    ClaimQueryFilter,
    DEFAULT_GRAPH_DEFINITION,
    DocumentManifest,
    GraphDefinition,
    GraphInboxItem,
    GraphMutationPlanRecord,
    GraphRuntime,
    ResourceStoreStatus,
    SectionManifest,
    build_graph_runtime,
    parse_resource_id,
    require_resource_slug,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.resources import (
    build_chunk,
    build_import_manifest,
    chunk_not_found,
    document_not_found,
    find_chunk_ref,
    read_source_document,
    section_not_found,
)
from potpie_context_engine.testing.conformance import (
    GraphBackendConformanceMixin,
    ResourceStoreConformanceMixin,
    run_graph_backend_conformance,
    run_resource_store_conformance,
    write_import_directory,
)


@dataclass(slots=True)
class InMemoryGraphPlanStore:
    _records: dict[tuple[str, str], GraphMutationPlanRecord] = field(
        default_factory=dict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, record: GraphMutationPlanRecord) -> None:
        with self._lock:
            self._records[(record.pot_id, record.plan_id)] = record

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        with self._lock:
            return self._records.get((pot_id, plan_id))

    def compare_and_set(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        key = (expected.pot_id, expected.plan_id)
        if key != (replacement.pot_id, replacement.plan_id):
            raise ValueError("plan compare-and-set cannot change plan identity")
        with self._lock:
            if self._records.get(key) != expected:
                return False
            self._records[key] = replacement
            return True

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        with self._lock:
            records = [
                record
                for (record_pot, _), record in self._records.items()
                if record_pot == pot_id
                and (plan_id is None or record.plan_id == plan_id)
                and (mutation_id is None or record.mutation_id == mutation_id)
                and (since is None or record.created_at >= since)
                and (until is None or record.created_at <= until)
            ]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return tuple(records[:limit] if limit is not None else records)

    async def save_async(self, record: GraphMutationPlanRecord) -> None:
        self.save(record)

    async def get_async(
        self, *, pot_id: str, plan_id: str
    ) -> GraphMutationPlanRecord | None:
        return self.get(pot_id=pot_id, plan_id=plan_id)

    async def compare_and_set_async(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        return self.compare_and_set(expected=expected, replacement=replacement)

    async def list_async(self, **kwargs):
        return self.list(**kwargs)


@dataclass(slots=True)
class InMemoryGraphInboxStore:
    _items: dict[tuple[str, str], GraphInboxItem] = field(default_factory=dict)

    def save(self, item: GraphInboxItem) -> None:
        self._items[(item.pot_id, item.item_id)] = item

    def get(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        return self._items.get((pot_id, item_id))

    def list(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphInboxItem, ...]:
        items = [
            item
            for (item_pot, _), item in self._items.items()
            if item_pot == pot_id
            and (not status or item.status in status)
            and (claimed_by is None or item.claimed_by == claimed_by)
            and (
                suspected_subgraph is None
                or suspected_subgraph in item.suspected_subgraphs
            )
            and (source_ref is None or source_ref in item.source_refs)
            and (since is None or item.created_at >= since)
            and (until is None or item.created_at <= until)
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(items[:limit] if limit is not None else items)

    async def save_async(self, item: GraphInboxItem) -> None:
        self.save(item)

    async def get_async(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        return self.get(pot_id=pot_id, item_id=item_id)

    async def list_async(self, **kwargs):
        return self.list(**kwargs)


@dataclass(frozen=True, slots=True)
class _StoredResource:
    manifest: DocumentManifest
    texts: dict[tuple[str, int], str]


@dataclass(slots=True)
class InMemoryResourceStore:
    """Reference ``ResourceStorePort`` with no filesystem behind it.

    Shares the import-directory validator with ``LocalResourceStore``, so the
    two answer the same contract. Atomicity comes for free: validation raises
    before any state is touched, leaving a prior revision intact. Pot isolation
    is structural — every document is keyed by ``(pot_id, doc)``.
    """

    _documents: dict[tuple[str, str], _StoredResource] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def import_dir(
        self,
        *,
        pot_id: str,
        slug: str,
        source_dir: Path,
        source_ref: str | None = None,
        source_kind: str | None = None,
    ) -> DocumentManifest:
        doc = require_resource_slug(slug, kind="document")
        source = read_source_document(Path(source_dir))
        with self._lock:
            stored = self._documents.get((pot_id, doc))
            manifest = build_import_manifest(
                pot_id=pot_id,
                doc=doc,
                source=source,
                prior=stored.manifest if stored else None,
                source_ref=source_ref,
                source_kind=source_kind,
            )
            self._documents[(pot_id, doc)] = _StoredResource(
                # The diff and warning fields report on one import; only the
                # structure is durable state.
                manifest=DocumentManifest(
                    pot_id=manifest.pot_id,
                    doc=manifest.doc,
                    revision=manifest.revision,
                    source_ref=manifest.source_ref,
                    source_kind=manifest.source_kind,
                    sections=manifest.sections,
                ),
                texts=dict(source.texts),
            )
        return manifest

    def get(self, *, pot_id: str, resource_id: str) -> Chunk:
        return self.get_many(pot_id=pot_id, resource_ids=(resource_id,))[0]

    def get_many(
        self, *, pot_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        with self._lock:
            for resource_id in resource_ids:
                resource = parse_resource_id(resource_id)
                stored = self._documents.get((pot_id, resource.doc))
                if stored is None:
                    raise chunk_not_found(resource_id)
                ref = find_chunk_ref(
                    stored.manifest, section=resource.section, seq=resource.seq
                )
                text = stored.texts.get((resource.section, resource.seq))
                if ref is None or text is None:
                    raise chunk_not_found(resource_id)
                chunks.append(
                    build_chunk(
                        manifest=stored.manifest, resource=resource, ref=ref, text=text
                    )
                )
        return tuple(chunks)

    def list(
        self, *, pot_id: str, slug: str, section: str | None = None
    ) -> tuple[SectionManifest, ...]:
        doc = require_resource_slug(slug, kind="document")
        with self._lock:
            stored = self._documents.get((pot_id, doc))
        if stored is None:
            raise document_not_found(doc)
        if section is None:
            return stored.manifest.sections
        wanted = require_resource_slug(section, kind="section")
        rows = tuple(row for row in stored.manifest.sections if row.slug == wanted)
        if not rows:
            raise section_not_found(doc, wanted)
        return rows

    def delete(self, *, pot_id: str, slug: str) -> bool:
        doc = require_resource_slug(slug, kind="document")
        with self._lock:
            return self._documents.pop((pot_id, doc), None) is not None

    def purge_pot(self, pot_id: str) -> bool:
        with self._lock:
            owned = [key for key in self._documents if key[0] == pot_id]
            for key in owned:
                del self._documents[key]
            return bool(owned)

    def status(self, *, pot_id: str | None = None) -> ResourceStoreStatus:
        with self._lock:
            documents = (
                sum(1 for key in self._documents if key[0] == pot_id)
                if pot_id
                else None
            )
        # Nothing to be unready about: the store is the process it runs in.
        return ResourceStoreStatus(
            kind="in_memory",
            ready=True,
            location=None,
            documents=documents,
            detail="resources are lost when the process exits",
        )


def build_test_backend(
    *, definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
) -> InMemoryGraphBackend:
    return InMemoryGraphBackend(definition=definition)


def build_test_graph_runtime(
    *, definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
) -> GraphRuntime:
    return build_graph_runtime(
        build_test_backend(definition=definition),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        definition,
    )


def assert_graph_backend_conforms(
    backend_factory: Callable[[], object],
) -> None:
    """Small dependency-free smoke suite reusable by third-party backends.

    The package's pytest conformance module layers the full behavioral suite
    over this startup/capability/isolation check.
    """

    backend = backend_factory().bind_definition(DEFAULT_GRAPH_DEFINITION)
    capabilities = backend.capabilities()
    if not capabilities.mutation or not capabilities.claim_query:
        raise AssertionError("backend must implement mutation and claim_query")
    first = backend.mutation.readiness("conformance:first")
    second = backend.mutation.readiness("conformance:second")
    if not first.ready or not second.ready:
        raise AssertionError("backend must be ready for two independent pots")
    backend.mutation.reset_pot("conformance:first")
    leaked = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="conformance:second")
    )
    if any(row.pot_id != "conformance:second" for row in leaked):
        raise AssertionError("backend claim queries are not pot-isolated")


__all__ = [
    "GraphBackendConformanceMixin",
    "InMemoryGraphBackend",
    "InMemoryGraphInboxStore",
    "InMemoryGraphPlanStore",
    "InMemoryResourceStore",
    "ResourceStoreConformanceMixin",
    "assert_graph_backend_conforms",
    "build_test_backend",
    "build_test_graph_runtime",
    "run_graph_backend_conformance",
    "run_resource_store_conformance",
    "write_import_directory",
]
