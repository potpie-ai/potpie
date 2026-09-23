"""``ResourceFacade`` — the store, the graph, and the index joined into one write.

Lives here rather than on ``host/shell`` because it is an application service,
not a host-shell concern, and because the managed service composes it *without*
``HostShell``: that module imports ``potpie.daemon`` for the local lifecycle,
which a shared server has no business depending on. ``host.shell`` re-exports
this name, so every existing import keeps working.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from potpie_context_core.ports.claim_query import ClaimQueryFilter, ClaimQueryPort
from potpie_context_core.ports.graph_service import GraphService
from potpie_context_core.ports.graph.snapshot import GraphSnapshotPort, SnapshotManifest
from potpie_context_core.ports.resource_index import (
    DEFAULT_DRAIN_BUDGET,
    DrainReport,
    IndexReport,
    ResourceIndexPort,
    ResourceIndexStatus,
)
from potpie_context_core.ports.resource_store import (
    RESOURCE_NOT_FOUND,
    Chunk,
    DocumentManifest,
    ImportFiles,
    ResourceStoreError,
    ResourceStorePort,
    ResourceStoreStatus,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
    ResourceBatchResult,
)
from potpie_context_core.resource_projection import project_chunk
from potpie_context_core.resource_to_semantic import (
    RESOURCE_SUBGRAPH,
    SCOPE_PREDICATE,
    SECTION_PREDICATE,
    DependentClaim,
    ResourceDeleteResult,
    ResourceImportResult,
    document_key,
    resource_delete_to_semantic_request,
    resource_import_to_semantic_request,
    section_key,
)
from potpie_context_core.semantic_mutations import (
    MutationActor,
    SemanticMutation,
    SemanticMutationRequest,
    SemanticMutationResult,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResourceFacade:
    """Holds the resource store behind the host, and joins it to the graph.

    Two jobs beyond passthrough. :meth:`import_dir` writes both halves of a
    document — bytes to the store, structure to the graph — because they are one
    user action and splitting them across two daemon calls would let the process
    die between them. :meth:`get` resolves ``--with-neighbors`` here for the same
    reason: a section's chunk list is needed to name the chunks either side of
    the ones asked for, and doing that CLI-side would cost a second round trip.
    Everything else is the port verbatim.

    ``claims`` is read-only and exists for two jobs the write alone cannot do:
    finding what *else* points at a section before it is removed, and reading a
    write back to see whether it actually landed.

    ``index`` is the third destination an import writes to. It is derived state
    over the same bytes, so it is rebuilt rather than repaired and a failure to
    write it never fails the import: losing search over a document is bad,
    losing the document is worse.
    """

    store: ResourceStorePort
    graph: GraphService | None = None
    claims: ClaimQueryPort | None = None
    index: ResourceIndexPort | None = None
    drain: Any = None
    """The background drain, when one is running, so writes can nudge it."""
    snapshot: GraphSnapshotPort | None = None

    def export_snapshot(self, *, pot_id: str) -> dict[str, Any]:
        """Export graph and document revisions as one portable bundle."""
        from .snapshot_archive import export_archive

        return export_archive(self, pot_id=pot_id)

    def import_snapshot(self, *, pot_id: str, payload: dict[str, Any]) -> SnapshotManifest:
        """Restore validated graph data and its original document text."""
        from .snapshot_archive import import_archive

        return import_archive(self, pot_id=pot_id, payload=payload)

    def import_dir(
        self,
        *,
        pot_id: str,
        slug: str,
        source_dir: Path | None = None,
        files: ImportFiles | None = None,
        source_ref: str | None = None,
        source_kind: str | None = None,
    ) -> ResourceImportResult:
        """Store the bytes, put the structure in the graph, index the text.

        Bytes first, deliberately: there is no transaction across the stores,
        and a failed graph write leaves orphan files the next import
        overwrites, whereas the reverse order would leave live claims citing
        chunks that do not exist. The index comes last for the same reason one
        step further — it is the only one of the three that can be recomputed
        from the others.
        """
        from .protocol_resources import protect_protocol_source

        prior = self._current_manifest(pot_id=pot_id, slug=slug)

        protect_protocol_source(
            self.store,
            self.claims,
            pot_id=pot_id,
            slug=slug,
            files=files,
            source_dir=source_dir,
        )
        manifest = self.store.import_dir(
            pot_id=pot_id,
            slug=slug,
            source_dir=source_dir,
            files=files,
            source_ref=source_ref,
            source_kind=source_kind,
        )
        index_report = self._index_document(pot_id=pot_id, manifest=manifest)
        manifest, review_claims, review_errors = self._review_claims_for_import(
            pot_id=pot_id, prior=prior, current=manifest
        )
        if self.graph is None:
            return ResourceImportResult(
                manifest=manifest,
                index=index_report,
                review_required_claim_keys=review_claims,
                review_marker_errors=review_errors,
            )
        mutation = self.graph.mutate(
            resource_import_to_semantic_request(
                manifest,
                dependent_claims=self._dependent_claims(
                    pot_id=pot_id,
                    section_keys=tuple(
                        section_key(manifest.doc, slug)
                        for slug in manifest.sections_removed
                    ),
                ),
            )
        )
        return ResourceImportResult(
            manifest=manifest,
            graph=mutation,
            missing_claim_keys=self._unreadable_claims(
                pot_id=pot_id, mutation=mutation
            ),
            # After the write, not before: this import's own retractions can
            # take the last scope edge with them when a linked section leaves.
            scope_claim_count=self._scope_claim_count(pot_id=pot_id, manifest=manifest),
            index=index_report,
            review_required_claim_keys=review_claims,
            review_marker_errors=review_errors,
        )

    def _current_manifest(self, *, pot_id: str, slug: str) -> DocumentManifest | None:
        try:
            return self.store.current_manifest(pot_id=pot_id, slug=slug)
        except ResourceStoreError as exc:
            if exc.code == RESOURCE_NOT_FOUND:
                return None
            raise

    def _review_claims_for_import(
        self,
        *,
        pot_id: str,
        prior: DocumentManifest | None,
        current: DocumentManifest,
    ) -> tuple[DocumentManifest, tuple[str, ...], tuple[str, ...]]:
        refs = current.pending_review_refs
        sections = current.pending_review_sections
        reason = current.pending_review_reason
        if prior is not None and prior.revision != current.revision:
            kept = set(current.sections_kept)
            changed_sections = tuple(
                section for section in prior.sections if section.slug not in kept
            )
            refs = tuple(
                dict.fromkeys(
                    (
                        *refs,
                        *(
                            ref
                            for section in changed_sections
                            for chunk in section.chunks
                            for ref in (
                                format_resource_id(
                                    prior.doc, section.slug, chunk.seq
                                ),
                                format_resource_id(
                                    prior.doc,
                                    section.slug,
                                    chunk.seq,
                                    revision=prior.revision,
                                ),
                            )
                        ),
                    )
                )
            )
            sections = tuple(
                dict.fromkeys((*sections, *(section.slug for section in changed_sections)))
            )
            reason = f"document {prior.doc!r} changed in revision {current.revision}"
        if not refs or not sections or not reason:
            return current, (), ()
        keys, errors = self._claim_keys_citing(
            pot_id=pot_id,
            refs=refs,
            doc=current.doc,
            sections=sections,
            reason=reason,
        )
        if not errors:
            cleared = self.store.clear_pending_review(
                pot_id=pot_id, slug=current.doc,
                expected_revision=current.revision, expected_refs=refs,
            )
            if cleared.revision == current.revision and not cleared.pending_review_refs:
                current = replace(
                    current,
                    pending_review_refs=(),
                    pending_review_sections=(),
                    pending_review_reason=None,
                )
            else:
                current = cleared
        return current, keys, errors

    def _claim_keys_citing(
        self,
        *,
        pot_id: str,
        refs: tuple[str, ...],
        doc: str,
        sections: tuple[str, ...],
        reason: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.claims is None or not refs:
            return (), ()
        rows = self.claims.find_claims(
            ClaimQueryFilter(pot_id=pot_id, include_invalidated=True)
        )
        affected = []
        section_set = set(sections)
        exact = set(refs)
        for row in rows:
            cited = tuple(
                dict.fromkeys(
                    (
                        *row.source_refs,
                        *(str(item.get("source_ref") or "") for item in row.evidence),
                    )
                )
            )
            matched = False
            for source_ref in cited:
                if source_ref in exact:
                    matched = True
                    break
                try:
                    parsed = parse_resource_id(source_ref)
                except ResourceStoreError:
                    continue
                if parsed.doc == doc and parsed.section in section_set:
                    matched = True
                    break
            if matched and row.predicate != SECTION_PREDICATE:
                affected.append(row)
        marker_errors = self._mark_claims_for_review(
            pot_id=pot_id,
            rows=affected,
            refs=tuple(dict.fromkeys(refs)),
            reason=reason,
        )
        keys = tuple(
            sorted(
                {
                    row.claim_key
                    for row in affected
                    if row.claim_key and row.predicate != SECTION_PREDICATE
                }
            )
        )
        return (() if marker_errors else keys), marker_errors

    def _mark_claims_for_review(
        self,
        *,
        pot_id: str,
        rows: list[Any],
        refs: tuple[str, ...],
        reason: str,
    ) -> tuple[str, ...]:
        """Persist evidence-review metadata without changing conclusion text."""
        if not rows:
            return ()
        if self.graph is None or self.claims is None:
            return (
                "evidence review markers were not persisted because no graph write service is wired",
            )
        keys = {key for row in rows for key in (row.subject_key, row.object_key)}
        labels = self.claims.entity_labels(pot_id=pot_id, entity_keys=keys)
        def concrete_type(key: str) -> str | None:
            concrete = sorted(label for label in labels.get(key, ()) if label != "Entity")
            return concrete[0] if concrete else None

        operations = []
        for row in rows:
            evidence = [dict(item) for item in row.evidence]
            if not evidence:
                evidence = [{"source_ref": ref} for ref in row.source_refs]
            raw = {
                "op": "assert_claim",
                "subgraph": row.subgraph or "knowledge",
                "predicate": row.predicate,
                "subject": {
                    "key": row.subject_key,
                    "type": concrete_type(row.subject_key),
                },
                "object": {
                    "key": row.object_key,
                    "type": concrete_type(row.object_key),
                },
                "truth": row.truth or "agent_claim",
                "confidence": row.confidence,
                "description": row.description or row.fact,
                "environment": row.environment,
                "valid_from": row.valid_at.isoformat() if row.valid_at else None,
                "valid_until": row.valid_until.isoformat() if row.valid_until else None,
                "observed_at": row.observed_at.isoformat() if row.observed_at else None,
                "evidence": evidence,
                "extra": {
                    "evidence_review_required": True,
                    "evidence_review_reason": reason,
                    "evidence_review_refs": list(refs),
                    "_preserve_claim_key": row.claim_key,
                    "_preserve_claim_properties": dict(row.properties),
                    "_preserve_claim_fact": row.fact,
                    "_preserve_source_system": row.source_system,
                    "_expected_claim_snapshot": _claim_snapshot(row),
                },
            }
            operations.append(SemanticMutation.parse(raw))
        review_digest = hashlib.sha256(
            json.dumps(
                [asdict(operation) for operation in operations],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        try:
            result = self.graph.mutate(
                SemanticMutationRequest(
                    pot_id=pot_id,
                    operations=tuple(operations),
                    idempotency_key=f"resource-evidence-review:{review_digest}",
                    created_by=MutationActor(
                        surface="resource", harness="resource_facade"
                    ),
                    allow_review_required=True,
                    approved_by="resource_evidence_lifecycle",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report partial lifecycle failure
            logger.exception("failed to persist evidence review markers")
            return (f"evidence review markers were not persisted: {exc}",)
        if not result.ok:
            errors = tuple(
                issue.message or issue.code for issue in result.issues
            ) or tuple(result.warnings) or (
                result.detail or "graph rejected evidence review markers",
            )
            logger.warning("failed to persist evidence review markers: %s", errors)
            return errors
        return ()

    def _index_document(
        self, *, pot_id: str, manifest: DocumentManifest
    ) -> IndexReport | None:
        """Feed one document's chunks to the index, or report why not.

        Reads the chunks back out of the store rather than keeping the import's
        own copy: the store is the source of truth, and indexing what was
        *stored* is the only version of this that cannot drift from what
        ``resource get`` will later return.

        Never raises. An index failure is reported through the result and the
        CLI turns it into a warning, because the bytes and the structure are
        already durable and the fix — ``resource index rebuild`` — needs no
        information this call has."""
        if self.index is None:
            return None
        try:
            resource_ids = tuple(
                format_resource_id(
                    manifest.doc,
                    section.slug,
                    ref.seq,
                    revision=manifest.revision,
                )
                for section in manifest.sections
                for ref in section.chunks
            )
            report = self.index.index_document(
                pot_id=pot_id,
                manifest=manifest,
                chunks=tuple(project_chunk(chunk) for chunk in self.store.get_many(pot_id=pot_id, resource_ids=resource_ids)),
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.warning("resource index write failed for %s: %s", manifest.doc, exc)
            return IndexReport(
                doc=manifest.doc,
                profile=_index_profile(self.index),
                detail=f"indexing failed, so this document is not searchable: {exc}",
            )
        if self.drain is not None and report.pending_embeddings:
            # A signal, not a handoff: the pending rows are already durable, so
            # the worst a lost wake-up costs is one idle interval.
            self.drain.signal()
        return report

    def _scope_claim_count(
        self, *, pot_id: str, manifest: DocumentManifest
    ) -> int | None:
        """How many live ``DOCUMENTS`` claims this document is reachable by.

        Both the document key and every section key, because the skills teach
        section-level links as the precise option — counting only the document
        would go on telling an agent to link a document whose every section is
        already linked, which is how the nudge became noise.

        ``None`` when no claim query is wired: the caller must say "unknown",
        never "unlinked".
        """
        if self.claims is None:
            return None
        subjects = (
            document_key(manifest.doc),
            *(section_key(manifest.doc, row.slug) for row in manifest.sections),
        )
        rows = self.claims.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                predicate_in=(SCOPE_PREDICATE,),
                subject_key_in=subjects,
            )
        )
        return len(rows)

    def _unreadable_claims(
        self, *, pot_id: str, mutation: SemanticMutationResult
    ) -> tuple[str, ...]:
        """Which of the claims this write reported are invisible to a read.

        ``status == "applied"`` counts *submitted* operations, not created rows,
        so a write that lands on an existing tombstone reports success and stays
        unreadable. Import is the one command where that is silent by design —
        the bytes are on disk either way — so it reads its own write back the
        way ``graph commit --verify`` does. One extra query per import, on a
        command that already did file I/O.
        """
        if self.claims is None or not mutation.ok or not mutation.claim_keys:
            return ()
        keys = tuple(mutation.claim_keys)
        live = {
            row.claim_key
            for row in self.claims.find_claims(
                ClaimQueryFilter(pot_id=pot_id, claim_key_in=keys)
            )
            if row.claim_key
        }
        return tuple(key for key in keys if key not in live)

    def get(
        self, *, pot_id: str, resource_ids: tuple[str, ...], with_neighbors: bool = False,
    ) -> tuple[Chunk, ...] | ResourceBatchResult:
        """One host read keeps successful roots and reports every failed id.

        Native stores read under a document lock with cached manifests. Older
        stores keep their existing all-success path; only failed reads use the
        compatibility per-root collector, entirely inside this host call.
        """
        from potpie_context_core.resource_reads import read_batch
        from potpie_context_core.ports.resource_store import (
            RESOURCE_GET_MAX_IDS, RESOURCE_BATCH_TOO_LARGE, RESOURCE_READ_BUDGET_EXCEEDED,
        )

        if len(resource_ids) > RESOURCE_GET_MAX_IDS:
            raise ResourceStoreError(
                RESOURCE_BATCH_TOO_LARGE,
                f"Resource batches support at most {RESOURCE_GET_MAX_IDS} ids; received {len(resource_ids)}. No chunks were read.",
                recommended_next_action=f"Split the original ids into batches of at most {RESOURCE_GET_MAX_IDS}.",
            )
        native = getattr(self.store, "get_batch", None)
        if callable(native):
            result = native(pot_id=pot_id, resource_ids=resource_ids, with_neighbors=with_neighbors)
        else:
            try:
                ids = self._with_neighbors(pot_id=pot_id, resource_ids=resource_ids) if with_neighbors else resource_ids
                return tuple(project_chunk(chunk) for chunk in self.store.get_many(pot_id=pot_id, resource_ids=ids))
            except ResourceStoreError:
                remaining_calls = 64

                def bounded_call(call, **kwargs):
                    nonlocal remaining_calls
                    if remaining_calls == 0:
                        raise ResourceStoreError(
                            RESOURCE_READ_BUDGET_EXCEEDED,
                            "This id was not fully read: the legacy store's 64-call recovery budget was reached. Follow up only unresolved ids in a smaller batch.",
                        )
                    remaining_calls -= 1
                    return call(**kwargs)

                result = read_batch(
                    resource_ids,
                    pot_id=pot_id,
                    read=lambda value: bounded_call(self.store.get, pot_id=pot_id, resource_id=value),
                    sections=lambda chunk: bounded_call(self.store.list, pot_id=pot_id, slug=chunk.doc, revision=chunk.revision),
                    with_neighbors=with_neighbors,
                )
        safe_result = replace(
            result, chunks=tuple(project_chunk(chunk) for chunk in result.chunks)
        )
        return safe_result.chunks if safe_result.status == "success" else safe_result

    def list(
        self, *, pot_id: str, slug: str, section: str | None = None
    ) -> tuple[SectionManifest, ...]:
        return self.store.list(pot_id=pot_id, slug=slug, section=section)

    def current_manifest(self, *, pot_id: str, slug: str) -> DocumentManifest:
        return self.store.current_manifest(pot_id=pot_id, slug=slug)

    def delete(self, *, pot_id: str, slug: str) -> ResourceDeleteResult:
        """Remove one document's bytes and retract every claim about it.

        Graph first, then bytes: a failed store delete leaves retracted claims
        (search misses; ``get`` still works until a retry), whereas bytes-first
        would leave search landing on chunk ids that no longer resolve.

        Returns both halves rather than a bare boolean so the caller can report
        what the *graph* did instead of inferring it from the store.
        """
        try:
            sections = self.store.list(pot_id=pot_id, slug=slug)
        except ResourceStoreError as exc:
            if getattr(exc, "code", None) == RESOURCE_NOT_FOUND:
                return ResourceDeleteResult(removed=False)
            raise
        from .protocol_resources import protect_protocol_source

        protect_protocol_source(self.store, self.claims, pot_id=pot_id, slug=slug)
        manifest = self.store.current_manifest(pot_id=pot_id, slug=slug)
        evidence_refs = tuple(
            ref
            for section in manifest.sections
            for chunk in section.chunks
            for ref in (
                format_resource_id(slug, section.slug, chunk.seq),
                format_resource_id(
                    slug,
                    section.slug,
                    chunk.seq,
                    revision=manifest.revision,
                ),
            )
        )
        reason = f"document {slug!r} was explicitly deleted"
        self.store.set_pending_review(
            pot_id=pot_id,
            slug=slug,
            refs=evidence_refs,
            sections=tuple(section.slug for section in manifest.sections),
            reason=reason,
            expected_revision=manifest.revision,
        )
        review_claims, review_errors = self._claim_keys_citing(
            pot_id=pot_id,
            refs=evidence_refs,
            doc=slug,
            sections=tuple(section.slug for section in manifest.sections),
            reason=reason,
        )
        if review_errors:
            return ResourceDeleteResult(
                removed=False,
                review_marker_errors=review_errors,
            )
        cleared = self.store.clear_pending_review(
            pot_id=pot_id, slug=slug, expected_revision=manifest.revision,
            expected_refs=evidence_refs,
        )
        if cleared.pending_review_refs:
            return ResourceDeleteResult(
                removed=False,
                review_marker_errors=("document changed while evidence review was being recorded; retry deletion",),
            )
        graph_result = None
        if self.graph is not None and sections:
            slugs = tuple(section.slug for section in sections)
            graph_result = self.graph.mutate(
                resource_delete_to_semantic_request(
                    pot_id=pot_id,
                    doc=slug,
                    section_slugs=slugs,
                    dependent_claims=self._dependent_claims(
                        pot_id=pot_id,
                        section_keys=tuple(section_key(slug, s) for s in slugs),
                    ),
                )
            )
        # Before the bytes go: a chunk id that still ranks in search but no
        # longer resolves is the one failure mode a caller cannot diagnose,
        # and dropping index rows for a document that then survives a failed
        # store delete only costs a rebuild.
        self._drop_from_index(pot_id=pot_id, slug=slug)
        return ResourceDeleteResult(
            removed=self.store.delete(pot_id=pot_id, slug=slug),
            graph=graph_result,
            review_required_claim_keys=review_claims,
            review_marker_errors=review_errors,
        )

    def _drop_from_index(self, *, pot_id: str, slug: str) -> None:
        if self.index is None:
            return
        try:
            self.index.drop_document(pot_id=pot_id, slug=slug)
        except Exception as exc:  # noqa: BLE001 - a delete must still delete
            logger.warning("resource index drop failed for %s: %s", slug, exc)

    def _dependent_claims(
        self, *, pot_id: str, section_keys: tuple[str, ...]
    ) -> tuple[DependentClaim, ...]:
        """Every live claim that references a section about to disappear.

        Both directions, because a dangling reference is a defect whichever end
        the section sits on: a ``DOCUMENTS`` claim *from* the section hands out
        chunk ids that no longer resolve, and a claim citing the section *as*
        evidence points at the same dead chunks. ``SECTION_OF`` is excluded —
        the request already retracts those, and emitting them twice would put
        two ops for one claim in a single plan.

        Degrades to no cleanup rather than blocking the delete when no claim
        query is wired: losing the bytes with stale claims behind them is the
        status quo, and refusing to delete would be worse.
        """
        if self.claims is None or not section_keys:
            return ()
        rows = [
            *self.claims.find_claims(
                ClaimQueryFilter(pot_id=pot_id, subject_key_in=section_keys)
            ),
            *self.claims.find_claims(
                ClaimQueryFilter(pot_id=pot_id, object_key_in=section_keys)
            ),
        ]
        found: dict[tuple[str, str, str], DependentClaim] = {}
        for row in rows:
            if row.predicate == SECTION_PREDICATE:
                continue
            identity = (row.predicate, row.subject_key, row.object_key)
            found.setdefault(
                identity,
                DependentClaim(
                    predicate=row.predicate,
                    subject_key=row.subject_key,
                    object_key=row.object_key,
                    # Retract in the subgraph the claim was written into, not
                    # the resource store's own.
                    subgraph=row.subgraph or RESOURCE_SUBGRAPH,
                ),
            )
        return tuple(found.values())

    def purge_pot(self, pot_id: str) -> bool:
        if self.index is not None:
            try:
                self.index.purge_pot(pot_id)
            except Exception as exc:  # noqa: BLE001 - a purge must still purge
                logger.warning("resource index purge failed for %s: %s", pot_id, exc)
        return self.store.purge_pot(pot_id)

    def status(self, *, pot_id: str | None = None) -> ResourceStoreStatus:
        return self.store.status(pot_id=pot_id)

    # --- index ---------------------------------------------------------------
    # The three verbs behind ``potpie resource index``. They live on the facade
    # rather than exposing the port directly because two of them need the
    # *store* as well: a rebuild re-derives from the bytes, which only the
    # facade holds both halves of.

    def index_status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        """Profile, capabilities, counts, and outstanding embeddings.

        Never raises, like every other diagnostic on this facade: a host with
        no index configured answers ``ready=False`` with the reason, because
        "the index is off" is the single most useful thing this command can say
        and an exception is the least useful way to say it."""
        if self.index is None:
            return ResourceIndexStatus(
                profile="none",
                ready=False,
                detail="no resource index is wired on this host",
            )
        return self.index.status(pot_id=pot_id)

    def index_build(
        self,
        *,
        pot_id: str | None = None,
        budget: int = DEFAULT_DRAIN_BUDGET,
        wait: bool = False,
    ) -> DrainReport:
        """Drain pending embeddings now, in the caller's thread.

        The same work the background loop does, run synchronously so a script
        can depend on it having finished. ``wait`` keeps draining until nothing
        is pending rather than stopping at one budget — the difference between
        "make progress" and "be done", which is what a CI step or a post-deploy
        hook actually needs."""
        if self.index is None:
            return DrainReport(profile="none", detail="no resource index is wired")
        report = self.index.drain(pot_id=pot_id, budget=budget)
        while wait and report.remaining and report.embedded:
            # Stop on ``embedded == 0`` as well as on an empty backlog: a batch
            # that embeds nothing and still reports work outstanding means the
            # embedder is failing, and looping on that would hang the command
            # instead of returning a report that shows it.
            report = self.index.drain(pot_id=pot_id, budget=budget)
        return report

    def index_rebuild(
        self, *, pot_id: str, doc: str | None = None
    ) -> tuple[IndexReport, ...]:
        """Drop and re-derive the index from the files, which are the truth.

        The whole recovery story for a derived store: there is no migration and
        no repair, so anything wrong with the index — a drifted posting, a
        profile switch, a half-written import — is fixed by computing it again.
        Scoped to one document when ``doc`` is given, because re-embedding a
        whole corpus to fix one document is a cost nobody should have to pay.
        """
        if self.index is None:
            return ()
        slugs = (doc,) if doc else self._indexable_documents(pot_id=pot_id)
        reports: list[IndexReport] = []
        for slug in slugs:
            self.index.drop_document(pot_id=pot_id, slug=slug)
            sections = self.store.list(pot_id=pot_id, slug=slug)
            manifest = self._current_manifest(pot_id=pot_id, slug=slug) or DocumentManifest(
                pot_id=pot_id,
                doc=slug,
                # The store's ``list`` returns sections, not the document's
                # revision or provenance. Those live on stored chunks, so the
                # revision is taken from the chunks below rather than guessed.
                revision=0,
                source_ref=None,
                source_kind=None,
                sections=sections,
            )
            chunks = self.store.get_many(
                pot_id=pot_id,
                resource_ids=tuple(
                    format_resource_id(
                        slug, section.slug, ref.seq,
                        revision=manifest.revision or None,
                    )
                    for section in sections
                    for ref in section.chunks
                ),
            )
            if chunks:
                manifest = replace(
                    manifest,
                    revision=chunks[0].revision,
                    source_ref=chunks[0].source_ref,
                )
            reports.append(
                self.index.index_document(
                    pot_id=pot_id, manifest=manifest,
                    chunks=tuple(project_chunk(chunk) for chunk in chunks),
                )
            )
        if self.drain is not None and any(r.pending_embeddings for r in reports):
            self.drain.signal()
        return tuple(reports)

    def _indexable_documents(self, *, pot_id: str) -> tuple[str, ...]:
        """Every document a rebuild should cover, from the store where possible.

        The store is asked first because it is the source of truth: a document
        whose bytes exist but which the index never saw is exactly what a
        rebuild is for, and asking the index would skip it. Falling back to the
        index's own list keeps the command useful against a store that cannot
        enumerate."""
        lister = getattr(self.store, "documents", None)
        if callable(lister):
            return tuple(lister(pot_id=pot_id))
        index_lister = getattr(self.index, "documents", None)
        if callable(index_lister):
            return tuple(index_lister(pot_id=pot_id))
        return ()

    def _with_neighbors(
        self, *, pot_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        expanded: list[str] = []
        seen: set[str] = set()
        sequences: dict[tuple[str, str, int | None], tuple[int, ...]] = {}
        for resource_id in resource_ids:
            resource = parse_resource_id(resource_id)
            key = (resource.doc, resource.section, resource.revision)
            if key not in sequences:
                sections = self.store.list(
                    pot_id=pot_id,
                    slug=resource.doc,
                    section=resource.section,
                    revision=resource.revision,
                )
                sequences[key] = tuple(
                    sorted(ref.seq for row in sections for ref in row.chunks)
                )
            seqs = sequences[key]
            if resource.seq not in seqs:
                # Let the store raise the not-found the caller expects, rather
                # than inventing one from a listing.
                neighborhood = (resource.seq,)
            else:
                position = seqs.index(resource.seq)
                neighborhood = seqs[max(position - 1, 0) : position + 2]
            for seq in neighborhood:
                candidate = format_resource_id(
                    resource.doc,
                    resource.section,
                    seq,
                    revision=resource.revision,
                )
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        return tuple(expanded)


def _index_profile(index: ResourceIndexPort | None) -> str:
    """The profile name for a report built when the index itself misbehaved."""
    try:
        return index.capabilities().profile if index is not None else "none"
    except Exception:  # noqa: BLE001 - naming the profile must not raise here
        return "unknown"


__all__ = ["ResourceFacade"]


def _claim_snapshot(row: Any) -> dict[str, Any]:
    def timestamp(value: Any) -> str | None:
        return value.isoformat() if value is not None else None

    return {
        "claim_key": row.claim_key,
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "fact": row.fact,
        "description": row.description,
        "source_system": row.source_system,
        "source_refs": list(row.source_refs),
        "evidence": [dict(item) for item in row.evidence],
        "properties": dict(row.properties),
        "valid_at": timestamp(row.valid_at),
        "invalid_at": timestamp(row.invalid_at),
        "valid_until": timestamp(row.valid_until),
        "observed_at": timestamp(row.observed_at),
        "truth": row.truth,
        "confidence": row.confidence,
        "environment": row.environment,
    }
