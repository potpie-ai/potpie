"""Local-profile ingestion: run config scanners over a working tree.

Bridges the existing :class:`ScanWorkingTreeUseCase` (rebuild plan P4) onto the
local :class:`GraphBackend.mutation` port so ``potpie ingest scan`` writes
deterministic claims through the same one-mutation path as ``record``. The use
case wants an async ``CanonicalWriterAdapter``; the local backend exposes the
sync ``GraphMutationPort.apply(ReconciliationPlan)``, so :class:`_MutationWriter`
adapts one onto the other (entities-then-edges become a single plan apply).

Deliberately local-scoped: the host enumerates the working tree directly from
disk (no GitHub fetch, no run-history store). The managed profile keeps its own
scanner wiring in ``bootstrap/ingestion_server`` + the async pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from application.services.config_scanner_registry import ConfigSourceScannerRegistry
from application.use_cases.scan_working_tree import (
    ScanWorkingTreeResult,
    ScanWorkingTreeUseCase,
    WorkingTreeFile,
)
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef
from domain.ports.graph.mutation import GraphMutationPort
from domain.reconciliation import ReconciliationPlan

# Config files worth scanning, by name/suffix. Kept conservative so a scan walks
# a real repo cheaply; scanners themselves decide via ``handles()`` what to emit.
_SCAN_GLOBS = (
    "CODEOWNERS",
    "*.yaml",
    "*.yml",
    "package.json",
    "go.mod",
    "requirements.txt",
    "pyproject.toml",
    "openapi.json",
    "openapi.yaml",
)
_MAX_FILE_BYTES = 512_000
_SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}


@dataclass(slots=True)
class _MutationWriter:
    """Adapt the sync ``GraphMutationPort`` to the async writer the use case wants.

    Each ``upsert_*`` call lands as one :class:`ReconciliationPlan` apply, so the
    scanner write goes through the same canonical mutation path as ``record``.
    """

    mutation: GraphMutationPort

    async def upsert_entities(
        self, pot_id: str, items: list[EntityUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        plan = ReconciliationPlan(
            event_ref=_event_ref(pot_id, provenance),
            summary="config-scanner entities",
            entity_upserts=list(items),
        )
        result = await asyncio.to_thread(
            self.mutation.apply, plan, expected_pot_id=pot_id
        )
        return result.mutation_summary.entity_upserts_applied

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int:
        if not items:
            return 0
        plan = ReconciliationPlan(
            event_ref=_event_ref(pot_id, provenance),
            summary="config-scanner claims",
            edge_upserts=list(items),
        )
        result = await asyncio.to_thread(
            self.mutation.apply, plan, expected_pot_id=pot_id
        )
        return result.mutation_summary.edge_upserts_applied


def _event_ref(pot_id: str, provenance: ProvenanceRef) -> EventRef:
    return EventRef(
        event_id=provenance.source_event_id,
        source_system=provenance.source_system or "config-scanner",
        pot_id=pot_id,
    )


@dataclass(slots=True)
class IngestService:
    """Local ``ingest scan``: walk a working tree, run scanners, write claims."""

    mutation: GraphMutationPort
    scanner_registry: ConfigSourceScannerRegistry

    def scan_path(
        self, *, pot_id: str, root: str, run_id: str, repo_name: str | None = None
    ) -> ScanWorkingTreeResult:
        """Scan the working tree rooted at ``root`` and write claims for ``pot_id``."""
        files = list(_read_working_tree(root, repo_name=repo_name))
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=self.scanner_registry,
            canonical_writer=_MutationWriter(mutation=self.mutation),
        )
        return asyncio.run(
            use_case.execute(pot_id=pot_id, run_id=run_id, working_tree=files)
        )


def _read_working_tree(root: str, *, repo_name: str | None) -> Iterable[WorkingTreeFile]:
    """Yield candidate config files under ``root`` (skipping vendored dirs)."""
    base = Path(root)
    if not base.exists():
        raise ValueError(f"working-tree root does not exist: {root}")
    seen: set[Path] = set()
    for pattern in _SCAN_GLOBS:
        for path in base.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            seen.add(path)
            yield WorkingTreeFile(
                path=str(path.relative_to(base)),
                content=content,
                repo_name=repo_name,
            )


__all__ = ["IngestService"]
