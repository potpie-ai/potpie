"""Transactional portable snapshots for Neo4j."""

from __future__ import annotations

import re
from typing import Any, Mapping

from potpie_context_core.graph_snapshot import normalize_snapshot_payload, validate_snapshot_merge
from potpie_context_core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.adapters.outbound.graph.cypher import _coerce_props_for_neo4j
from potpie_context_engine.adapters.outbound.graph.native_snapshot import (
    claim_row, entity_row, imported_manifest, payload_from_rows, read_payload,
    stored_claim, stored_entity, write_payload,
    validate_native_snapshot_properties,
)

_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_REVISION_CONSTRAINT = (
    "CREATE CONSTRAINT potpie_revision_unique IF NOT EXISTS "
    "FOR (r:PotpieRevision) REQUIRE r.pot_id IS UNIQUE"
)
_SNAPSHOT = """
OPTIONAL MATCH (e:Entity {group_id:$pot})
WITH collect(CASE WHEN e IS NULL THEN null ELSE
  {key:e.entity_key,labels:labels(e),props:properties(e)} END) AS entities
OPTIONAL MATCH (:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(:Entity {group_id:$pot})
RETURN entities,collect(CASE WHEN r IS NULL THEN null ELSE properties(r) END) AS claims
"""


class Neo4jSnapshot:
    def __init__(self, settings: Any) -> None:
        self._settings = settings

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

    def import_data(self, *, pot_id: str, payload: Mapping[str, Any]) -> SnapshotManifest:
        incoming = normalize_snapshot_payload(payload, target_pot_id=pot_id)
        validate_native_snapshot_properties(incoming)
        _validate_labels(incoming)
        driver = self._driver()
        try:
            with driver.session() as session:
                session.execute_write(_ensure_revision_constraint)
                session.execute_write(lambda tx: _merge(tx, pot_id, incoming))
        finally:
            driver.close()
        return imported_manifest("memory", incoming, pot_id=pot_id)

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        return write_payload(destination, self.export_data(pot_id=pot_id))

    def import_(self, *, pot_id: str, source: str) -> SnapshotManifest:
        result = self.import_data(pot_id=pot_id, payload=read_payload(source))
        return SnapshotManifest(
            pot_id=result.pot_id, location=str(source), format_version=result.format_version,
            entity_count=result.entity_count, claim_count=result.claim_count,
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
        entity_rows=(entity_row(item["key"], item["labels"], item["props"]) for item in entities),
        claim_rows=(claim_row(props) for props in claims),
    )


def _ensure_revision_constraint(tx: Any) -> None:
    tx.run(_REVISION_CONSTRAINT).consume()


def _merge(tx: Any, pot_id: str, incoming: Mapping[str, Any]) -> None:
    # This read-dependent update takes Neo4j's node write lock. Validation and
    # every canonical write below consequently share one serializable pot gate.
    revision = tx.run(
        "MERGE (v:PotpieRevision {pot_id:$pot}) ON CREATE SET v.version=0,v.lock_sequence=0 "
        "SET v.lock_sequence=v.lock_sequence+1 RETURN v.version",
        pot=pot_id,
    ).single()
    if revision is None:
        raise RuntimeError("failed to lock Neo4j pot revision")
    existing = _read(tx, pot_id)
    validate_snapshot_merge(
        existing_entities=existing["entities"],
        existing_claims=existing["claims"],
        incoming=incoming,
    )
    existing_entity_keys = {row["key"] for row in existing["entities"]}
    existing_claim_keys = {row["claim_key"] for row in existing["claims"]}
    new_entities = [row for row in incoming["entities"] if row["key"] not in existing_entity_keys]
    new_claims = [row for row in incoming["claims"] if row["claim_key"] not in existing_claim_keys]
    if not new_entities and not new_claims:
        return
    for entity in new_entities:
        key, labels, props = stored_entity(entity, pot_id)
        suffix = "".join(f":{label}" for label in labels)
        tx.run(
            "MERGE (e:Entity {group_id:$pot,entity_key:$key}) SET e += $props "
            f"SET e{suffix}",
            pot=pot_id, key=key, props=_coerce_props_for_neo4j(props),
        ).consume()
    for claim in new_claims:
        props = stored_claim(claim, pot_id)
        tx.run(
            "MATCH (a:Entity {group_id:$pot,entity_key:$subject}),"
            "(b:Entity {group_id:$pot,entity_key:$object}) "
            "MERGE (a)-[r:RELATES_TO {group_id:$pot,claim_key:$claim_key}]->(b) "
            "SET r += $props",
            pot=pot_id,
            subject=claim["subject_key"],
            object=claim["object_key"],
            claim_key=claim["claim_key"],
            props=_coerce_props_for_neo4j(props),
        ).consume()
    tx.run(
        "MATCH (v:PotpieRevision {pot_id:$pot}) SET v.version=v.version+1",
        pot=pot_id,
    ).consume()


def _validate_labels(payload: Mapping[str, Any]) -> None:
    for entity in payload["entities"]:
        labels = tuple(entity["labels"])
        if any(not _LABEL.fullmatch(label) for label in labels):
            raise ValueError(f"invalid snapshot entity label: {labels!r}")


__all__ = ["Neo4jSnapshot"]
