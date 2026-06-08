"""Scan a pot's working tree through every registered config scanner.

Rebuild plan P4: scanners are deterministic, host-triggered, and write
through the canonical Position B writer with
``evidence_strength="deterministic"``. This use case is the seam
between the host's working-tree fetch and the scanner registry's
output:

1. Host enumerates a pot's working tree (commit-on-main or scheduled
   walk) and hands back ``(path, content)`` tuples.
2. Use case wraps each tuple in a :class:`ConfigFileRef`, dispatches
   through :class:`ConfigSourceScannerRegistry`.
3. Use case translates ``ScannerEntity`` → :class:`EntityUpsert` and
   ``ScannerClaim`` → :class:`EdgeUpsert`, then writes through the
   canonical writer with a deterministic-source :class:`ProvenanceRef`.

Failure isolation: a single scanner raising does not abort the batch —
the registry traps and records a warning. The use case still completes
and reports the warnings. The caller decides whether to surface them.

The use case is intentionally *not* tied to a transport: CLI, HTTP, or
a future commit-webhook can drive it. Inputs are flat data; outputs are
counts + warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Protocol

from application.services.config_scanner_registry import ConfigSourceScannerRegistry
from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef
from domain.ports.config_scanner import ConfigFileRef, ScannerClaim, ScannerEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkingTreeFile:
    """One host-supplied (path, content) pair the scanner pipeline consumes."""

    path: str
    content: str
    repo_name: str | None = None
    commit_sha: str | None = None


@dataclass(slots=True)
class ScanWorkingTreeResult:
    """Aggregated outcome from one ``scan_working_tree`` invocation."""

    entities_upserted: int = 0
    edges_upserted: int = 0
    scanners_run: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    skipped_files: int = 0


class CanonicalWriterAdapter(Protocol):
    """The minimal slice of the canonical writer this use case needs.

    Kept narrow so tests can fake the adapter without standing up Neo4j.
    """

    async def upsert_entities(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int: ...

    async def upsert_edges(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int: ...


@dataclass(slots=True)
class ScanWorkingTreeUseCase:
    """Run all registered scanners over a working tree, then write through canonical writer."""

    scanner_registry: ConfigSourceScannerRegistry
    canonical_writer: CanonicalWriterAdapter
    actor_id: str = "config-scanner-registry"

    async def execute(
        self,
        *,
        pot_id: str,
        run_id: str,
        working_tree: Iterable[WorkingTreeFile],
        observed_at: datetime | None = None,
        extra_provenance: Mapping[str, str] | None = None,
    ) -> ScanWorkingTreeResult:
        """Scan + write. ``run_id`` becomes the source_event_id on provenance.

        ``extra_provenance`` lets callers add ``source_kind``, ``source_ref``,
        or ``reconciliation_run_id`` without forcing the use case to know
        about transport-specific values.
        """
        if not pot_id:
            raise ValueError("scan_working_tree requires a non-empty pot_id")
        if not run_id:
            raise ValueError("scan_working_tree requires a non-empty run_id")

        observed_at = observed_at or datetime.now(tz=timezone.utc)
        warnings: list[str] = []
        skipped = 0

        file_refs = []
        for wt in working_tree:
            if not wt.path or not isinstance(wt.path, str):
                skipped += 1
                continue
            file_refs.append(
                ConfigFileRef(
                    path=wt.path,
                    content=wt.content,
                    repo_name=wt.repo_name,
                    commit_sha=wt.commit_sha,
                    observed_at=observed_at,
                )
            )

        if not file_refs:
            logger.info("scan_working_tree pot=%s: no files to scan", pot_id)
            return ScanWorkingTreeResult(skipped_files=skipped)

        aggregated = self.scanner_registry.scan_files(file_refs)
        warnings.extend(aggregated.warnings)

        if not aggregated.scanners_run:
            logger.info(
                "scan_working_tree pot=%s: no scanner matched any of %d files",
                pot_id,
                len(file_refs),
            )
            return ScanWorkingTreeResult(
                warnings=tuple(warnings),
                skipped_files=skipped,
            )

        provenance = ProvenanceRef(
            pot_id=pot_id,
            source_event_id=run_id,
            source_system="config-scanner",
            source_kind=(
                extra_provenance.get("source_kind")
                if extra_provenance
                else "working-tree-scan"
            ),
            source_ref=(
                extra_provenance.get("source_ref") if extra_provenance else None
            ),
            event_occurred_at=observed_at,
            event_received_at=observed_at,
            graph_updated_at=observed_at,
            created_by_agent=self.actor_id,
            reconciliation_run_id=(
                extra_provenance.get("reconciliation_run_id")
                if extra_provenance
                else None
            ),
        )

        entity_upserts = _to_entity_upserts(aggregated.entities)
        edge_upserts = _to_edge_upserts(aggregated.claims)

        # Order matters: entities before edges so the MERGE keys resolve.
        entities_written = 0
        edges_written = 0
        if entity_upserts:
            entities_written = await self.canonical_writer.upsert_entities(
                pot_id, entity_upserts, provenance
            )
        if edge_upserts:
            edges_written = await self.canonical_writer.upsert_edges(
                pot_id, edge_upserts, provenance
            )

        return ScanWorkingTreeResult(
            entities_upserted=entities_written,
            edges_upserted=edges_written,
            scanners_run=tuple(aggregated.scanners_run),
            warnings=tuple(warnings),
            skipped_files=skipped,
        )


# ---------------------------------------------------------------------------
# Pure translation helpers
# ---------------------------------------------------------------------------


def _to_entity_upserts(entities: Iterable[ScannerEntity]) -> list[EntityUpsert]:
    """Dedupe by entity_key, last-write-wins on properties."""
    by_key: dict[str, EntityUpsert] = {}
    for ent in entities:
        if not ent.entity_key:
            continue
        labels: tuple[str, ...] = ("Entity", ent.label) if ent.label else ("Entity",)
        props: dict[str, object] = dict(ent.properties)
        if ent.name and "name" not in props:
            props["name"] = ent.name
        existing = by_key.get(ent.entity_key)
        if existing is None:
            by_key[ent.entity_key] = EntityUpsert(
                entity_key=ent.entity_key,
                labels=labels,
                properties=props,
            )
        else:
            merged_props = dict(existing.properties)
            merged_props.update(props)
            merged_labels = tuple(dict.fromkeys((*existing.labels, *labels)))
            by_key[ent.entity_key] = EntityUpsert(
                entity_key=ent.entity_key,
                labels=merged_labels,
                properties=merged_props,
            )
    return list(by_key.values())


def _to_edge_upserts(claims: Iterable[ScannerClaim]) -> list[EdgeUpsert]:
    """Translate :class:`ScannerClaim` into the canonical-writer's edge shape."""
    out: list[EdgeUpsert] = []
    for claim in claims:
        if not (claim.subject_key and claim.object_key and claim.predicate):
            continue
        props: dict[str, object] = dict(claim.properties)
        # Reserved keys the canonical writer reads from properties dict
        props["source_ref"] = claim.source_ref
        props["source_system"] = claim.source_system
        props["evidence_strength"] = claim.evidence_strength
        props["fact"] = claim.fact
        props["valid_at"] = claim.valid_at
        out.append(
            EdgeUpsert(
                edge_type=claim.predicate,
                from_entity_key=claim.subject_key,
                to_entity_key=claim.object_key,
                properties=props,
            )
        )
    return out


__all__ = [
    "CanonicalWriterAdapter",
    "ScanWorkingTreeResult",
    "ScanWorkingTreeUseCase",
    "WorkingTreeFile",
]
