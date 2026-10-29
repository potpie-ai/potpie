"""Transactional portable snapshots for Neo4j."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.adapters.outbound.graph.native_snapshot import (
    claim_row,
    entity_row,
    imported_manifest,
    payload_from_rows,
    read_payload,
    write_payload,
)

_SNAPSHOT = """
OPTIONAL MATCH (e:Entity {group_id:$pot})
WITH collect(CASE WHEN e IS NULL THEN null ELSE
  {key:e.entity_key,labels:labels(e),props:properties(e)} END) AS entities
OPTIONAL MATCH (:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(:Entity {group_id:$pot})
RETURN entities,collect(CASE WHEN r IS NULL THEN null ELSE properties(r) END) AS claims
"""


class Neo4jSnapshot:
    def __init__(self, settings: Any, *, mutation: Any = None) -> None:
        self._settings = settings
        self._mutation = mutation

    def _mutation_port(self) -> Any:
        if self._mutation is None:
            from .backends.neo4j_backend import Neo4jGraphBackend

            self._mutation = Neo4jGraphBackend(self._settings).mutation
        return self._mutation

    def _driver(self) -> Any:
        from neo4j import GraphDatabase

        uri = self._settings.neo4j_uri()
        user = self._settings.neo4j_user()
        password = self._settings.neo4j_password()
        if not uri or user is None or password is None:
            raise RuntimeError("neo4j_unavailable")
        return GraphDatabase.driver(uri, auth=(user, password))

    def export_data(self, *, pot_id: str) -> dict[str, Any]:
        driver = self._driver()
        try:
            with driver.session() as session:
                return session.execute_read(lambda tx: _read(tx, pot_id))
        finally:
            driver.close()

    def import_data(
        self, *, pot_id: str, payload: Mapping[str, Any], mutation_id: str | None = None
    ) -> SnapshotManifest:
        mutation = self._mutation_port()
        if not mutation.atomic_mutations_supported:
            raise NotImplementedError(
                "Neo4j snapshot import requires native atomic mutations"
            )
        version = mutation.current_version(pot_id)
        batch, provenance = prepare_snapshot_import(
            pot_id=pot_id,
            payload=payload,
            expected_version=version,
            mutation_id=mutation_id,
        )
        result = mutation.compare_and_apply(
            batch,
            expected_pot_id=pot_id,
            expected_version=version,
            provenance_context=provenance,
        )
        return replace(
            imported_manifest("memory", batch.snapshot_import, pot_id=pot_id),
            metadata={
                "mutation_id": result.mutation_id,
                "execution": result.execution.to_dict() if result.execution else None,
                "administrative_summary": dict(result.administrative_summary),
            },
        )

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


def _read(tx: Any, pot_id: str) -> dict[str, Any]:
    row = tx.run(_SNAPSHOT, pot=pot_id).single()
    if row is None:
        entities, claims = [], []
    else:
        entities = [item for item in (row["entities"] or ()) if item is not None]
        claims = [item for item in (row["claims"] or ()) if item is not None]
    return payload_from_rows(
        pot_id=pot_id,
        entity_rows=(
            entity_row(item["key"], item["labels"], item["props"]) for item in entities
        ),
        claim_rows=(claim_row(props) for props in claims),
    )


__all__ = ["Neo4jSnapshot"]
