"""Atomic portable snapshots for FalkorDB and FalkorDBLite."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from redis.exceptions import WatchError

from potpie_context_engine.adapters.outbound.graph.native_snapshot import (
    claim_row,
    entity_row,
    imported_manifest,
    payload_from_rows,
    read_payload,
    write_payload,
)

from .falkordb_connection import transaction_client

_SNAPSHOT = (
    "OPTIONAL MATCH (e:Entity {group_id:$pot}) WHERE coalesce(e.retired,false)=false "
    "WITH collect(CASE WHEN e IS NULL THEN null ELSE [e.entity_key,labels(e),properties(e)] END) AS entities "
    "OPTIONAL MATCH (a:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(b:Entity {group_id:$pot}) "
    "WHERE coalesce(r.retired,false)=false AND coalesce(a.retired,false)=false AND coalesce(b.retired,false)=false "
    "RETURN entities,collect(properties(r))"
)


def _snapshot_rows(graph, pipe, pot_id):
    from .falkordb_journal import _read

    rows = _read(graph, pipe, _SNAPSHOT, {"pot": pot_id})
    return rows[0] if rows else ([], [])


def read_snapshot(graph, pipe, pot_id):
    entities, claims = _snapshot_rows(graph, pipe, pot_id)
    return payload_from_rows(
        pot_id=pot_id,
        entity_rows=(entity_row(key, labels, props) for key, labels, props in entities),
        claim_rows=(claim_row(props) for props in claims),
    )


class FalkorDBSnapshot:
    def __init__(self, graph_provider: Any, mutation: Any) -> None:
        self._graph_provider = graph_provider
        self._mutation = mutation

    def _graph(self) -> Any:
        return self._graph_provider()

    def export_data(self, *, pot_id: str) -> dict[str, Any]:
        graph = self._graph()
        while True:
            pipe = transaction_client(graph).pipeline()
            try:
                pipe.watch(graph.name)
                from .falkordb_revision import read_revision

                read_revision(graph, pot_id, pipe)
                entities, claims = _snapshot_rows(graph, pipe, pot_id)
                from .falkordb_journal import _state
                from .journal_snapshot_policy import require_snapshot_identity

                if _state(graph, pot_id, pipe) is not None:
                    require_snapshot_identity(
                        props.get("claim_key") for props in claims
                    )
                pipe.multi()
                pipe.execute()
                return payload_from_rows(
                    pot_id=pot_id,
                    entity_rows=(
                        entity_row(key, labels, props)
                        for key, labels, props in entities
                    ),
                    claim_rows=(claim_row(props) for props in claims),
                )
            except WatchError:
                continue
            finally:
                pipe.reset()

    def import_data(
        self, *, pot_id: str, payload: Mapping[str, Any], mutation_id: str | None = None
    ) -> SnapshotManifest:
        from potpie_context_engine.core.errors import GraphMutationVersionConflict
        from potpie_context_engine.core.graph_snapshot_mutation import (
            prepare_snapshot_import,
        )
        from .falkordb_admin import validate_administration

        for attempt in range(3):
            version = self._mutation.current_version(pot_id)
            batch, context = prepare_snapshot_import(
                pot_id=pot_id,
                payload=payload,
                expected_version=version,
                mutation_id=mutation_id,
            )
            validate_administration(batch)
            try:
                result = self._mutation.compare_and_apply(
                    batch,
                    expected_pot_id=pot_id,
                    expected_version=version,
                    provenance_context=context,
                )
            except GraphMutationVersionConflict:
                if attempt == 2:
                    raise
            else:
                manifest = imported_manifest(
                    "memory", batch.snapshot_import, pot_id=pot_id
                )
                return SnapshotManifest(
                    pot_id=manifest.pot_id,
                    location=manifest.location,
                    format_version=manifest.format_version,
                    entity_count=manifest.entity_count,
                    claim_count=manifest.claim_count,
                    metadata={
                        "mutation_id": result.mutation_id,
                        "execution": result.execution.to_dict()
                        if result.execution
                        else None,
                    },
                )
        raise AssertionError("unreachable snapshot preparation state")

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        return write_payload(destination, self.export_data(pot_id=pot_id))

    def import_(
        self, *, pot_id: str, source: str, mutation_id: str | None = None
    ) -> SnapshotManifest:
        result = self.import_data(
            pot_id=pot_id, payload=read_payload(source), mutation_id=mutation_id
        )
        return SnapshotManifest(
            pot_id=result.pot_id,
            location=str(source),
            format_version=result.format_version,
            entity_count=result.entity_count,
            claim_count=result.claim_count,
            metadata=result.metadata,
        )


__all__ = ["FalkorDBSnapshot"]
