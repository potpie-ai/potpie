"""Coordinate portable graph snapshots and resource text on the serving host."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.graph_snapshot import normalize_snapshot_payload, validate_snapshot_merge
from potpie_context_core.ports.graph.snapshot import SnapshotManifest
from potpie_context_core.resource_snapshot import normalize_resource_snapshot

if TYPE_CHECKING:
    from .resource_facade import ResourceFacade

logger = logging.getLogger(__name__)


def export_archive(facade: ResourceFacade, *, pot_id: str) -> dict[str, Any]:
    if facade.snapshot is None:
        raise CapabilityNotImplemented("resources.export_snapshot")
    exporter = getattr(facade.store, "export_snapshot", None)
    if not callable(exporter):
        raise CapabilityNotImplemented("resources.snapshot")
    payload = dict(facade.snapshot.export_data(pot_id=pot_id))
    payload["resources"] = exporter(pot_id=pot_id)
    return payload


def import_archive(
    facade: ResourceFacade, *, pot_id: str, payload: dict[str, Any]
) -> SnapshotManifest:
    if facade.snapshot is None:
        raise CapabilityNotImplemented("resources.import_snapshot")
    graph = normalize_snapshot_payload(
        {key: value for key, value in payload.items() if key != "resources"},
        target_pot_id=pot_id,
    )
    resources = normalize_resource_snapshot(payload.get("resources"))
    restorer = getattr(facade.store, "restore_snapshot", None)
    if not callable(restorer):
        raise CapabilityNotImplemented("resources.snapshot")
    current = facade.snapshot.export_data(pot_id=pot_id)
    validate_snapshot_merge(
        existing_entities=current["entities"],
        existing_claims=current["claims"],
        incoming=graph,
    )
    # Resource validation/conflict checking happens before publication. A graph
    # failure restores the prior resource tree; graph adapters commit atomically.
    with restorer(pot_id=pot_id, payload=resources):
        manifest = facade.snapshot.import_data(pot_id=pot_id, payload=graph)
    documents = {path.split("/", 1)[0] for path in resources["files"]}
    warnings: list[str] = []
    if facade.index is not None:
        try:
            facade.index_rebuild(pot_id=pot_id)
        except Exception:
            logger.exception("Snapshot restored; resource search index rebuild failed")
            warnings.append("Document text restored; run potpie resource index rebuild to restore search.")
    return replace(
        manifest,
        metadata={**manifest.metadata, "documents": len(documents), "warnings": warnings},
    )
