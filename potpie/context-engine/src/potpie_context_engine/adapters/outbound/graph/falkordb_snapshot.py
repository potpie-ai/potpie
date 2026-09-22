"""Atomic portable snapshots for FalkorDB and FalkorDBLite."""

from __future__ import annotations

import re
from typing import Any, Mapping

from redis.exceptions import ResponseError, WatchError

from potpie_context_core.graph_snapshot import normalize_snapshot_payload, validate_snapshot_merge
from potpie_context_core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.adapters.outbound.graph.cypher import _coerce_props_for_neo4j
from potpie_context_engine.adapters.outbound.graph.falkordb_atomic import _ensure_state, current_version
from potpie_context_engine.adapters.outbound.graph.native_snapshot import (
    claim_row, entity_row, imported_manifest, payload_from_rows, read_payload,
    stored_claim, stored_entity, write_payload,
    validate_native_snapshot_properties,
)

_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ENTITIES = "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key,labels(e),properties(e) ORDER BY e.entity_key"
_CLAIMS = "MATCH (:Entity {group_id:$pot})-[r:RELATES_TO {group_id:$pot}]->(:Entity {group_id:$pot}) RETURN properties(r) ORDER BY r.claim_key,r.name,r.subject_key,r.object_key,r.source_ref"


class FalkorDBSnapshot:
    def __init__(self, graph_provider: Any) -> None:
        self._graph_provider = graph_provider

    def _graph(self) -> Any:
        return self._graph_provider()

    def export_data(self, *, pot_id: str) -> dict[str, Any]:
        graph = self._graph()
        while True:
            pipe = graph.client.pipeline()
            try:
                pipe.watch(graph.name)
                try:
                    entities = graph.ro_query(_ENTITIES, params={"pot": pot_id}).result_set
                    claims = graph.ro_query(_CLAIMS, params={"pot": pot_id}).result_set
                except ResponseError as exc:
                    if "empty key" not in str(exc).lower():
                        raise
                    entities, claims = [], []
                pipe.multi()
                pipe.execute()
                return payload_from_rows(
                    pot_id=pot_id,
                    entity_rows=(entity_row(key, labels, props) for key, labels, props in entities),
                    claim_rows=(claim_row(props) for (props,) in claims),
                )
            except WatchError:
                continue
            finally:
                pipe.reset()

    def import_data(self, *, pot_id: str, payload: Mapping[str, Any]) -> SnapshotManifest:
        incoming = normalize_snapshot_payload(payload, target_pot_id=pot_id)
        validate_native_snapshot_properties(incoming)
        _validate_labels(incoming)
        graph = self._graph()
        existing = self.export_data(pot_id=pot_id)
        validate_snapshot_merge(
            existing_entities=existing["entities"],
            existing_claims=existing["claims"],
            incoming=incoming,
        )
        if _contains(existing, incoming):
            return imported_manifest("memory", incoming, pot_id=pot_id)
        _ensure_state(graph, pot_id)
        while True:
            pipe = graph.client.pipeline()
            try:
                pipe.watch(graph.name)
                before = current_version(graph, pot_id)
                existing = self.export_data(pot_id=pot_id)
                validate_snapshot_merge(
                    existing_entities=existing["entities"],
                    existing_claims=existing["claims"],
                    incoming=incoming,
                )
                if _contains(existing, incoming):
                    return imported_manifest("memory", incoming, pot_id=pot_id)
                query, params = _import_query(incoming, pot_id=pot_id, expected=before)
                pipe.multi()
                pipe.execute_command(
                    "GRAPH.QUERY", graph.name,
                    graph._build_params_header(params) + query,
                    "--compact",
                )
                pipe.execute()
                return imported_manifest("memory", incoming, pot_id=pot_id)
            except WatchError:
                continue
            finally:
                pipe.reset()

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        return write_payload(destination, self.export_data(pot_id=pot_id))

    def import_(self, *, pot_id: str, source: str) -> SnapshotManifest:
        result = self.import_data(pot_id=pot_id, payload=read_payload(source))
        return SnapshotManifest(
            pot_id=result.pot_id, location=str(source), format_version=result.format_version,
            entity_count=result.entity_count, claim_count=result.claim_count,
            metadata=result.metadata,
        )


def _import_query(payload: Mapping[str, Any], *, pot_id: str, expected: int) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {"pot": pot_id, "expected": expected}
    clauses = ["MATCH (v:PotpieRevision {pot_id:$pot}) WHERE v.version=$expected"]
    for index, entity in enumerate(payload["entities"]):
        key, labels, props = stored_entity(entity, pot_id)
        params[f"ek{index}"] = key
        params[f"ep{index}"] = _coerce_props_for_neo4j(props)
        suffix = "".join(f":{label}" for label in labels)
        clauses.append(
            f"CALL {{ WITH v MERGE (e:Entity {{group_id:$pot,entity_key:$ek{index}}}) "
            f"SET e += $ep{index} SET e{suffix} RETURN count(e) AS ec{index} }}"
        )
    for index, claim in enumerate(payload["claims"]):
        props = stored_claim(claim, pot_id)
        embedding = props.pop("fact_embedding", None)
        params[f"cp{index}"] = _coerce_props_for_neo4j(props)
        params[f"cs{index}"] = claim["subject_key"]
        params[f"co{index}"] = claim["object_key"]
        params[f"ck{index}"] = claim["claim_key"]
        vector_set = ""
        if embedding is not None:
            params[f"ce{index}"] = embedding
            vector_set = f"SET r.fact_embedding=vecf32($ce{index}) "
        clauses.append(
            f"CALL {{ WITH v MATCH (a:Entity {{group_id:$pot,entity_key:$cs{index}}}),"
            f"(b:Entity {{group_id:$pot,entity_key:$co{index}}}) "
            f"MERGE (a)-[r:RELATES_TO {{group_id:$pot,claim_key:$ck{index}}}]->(b) "
            f"SET r += $cp{index} {vector_set}RETURN count(r) AS cc{index} }}"
        )
    clauses.append("SET v.version=v.version+1 RETURN v.version")
    return " ".join(clauses), params


def _validate_labels(payload: Mapping[str, Any]) -> None:
    for entity in payload["entities"]:
        labels = tuple(entity["labels"])
        if any(not _LABEL.fullmatch(label) for label in labels):
            raise ValueError(f"invalid snapshot entity label: {labels!r}")


def _contains(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> bool:
    entity_keys = {row["key"] for row in existing["entities"]}
    claim_keys = {row["claim_key"] for row in existing["claims"]}
    return all(row["key"] in entity_keys for row in incoming["entities"]) and all(
        row["claim_key"] in claim_keys for row in incoming["claims"]
    )


__all__ = ["FalkorDBSnapshot"]
