"""Native Ladybug records for the shared local graph transaction semantics.

Identifiers, claim fields, embeddings and endpoints are native columns. JSON is
used only for open property maps and the existing closed receipt/journal codecs.
"""

from __future__ import annotations

import json
from typing import Any

from ._local_json_atomic import strict_json_load

CLAIM_COLUMNS = {
    **{
        name: "STRING"
        for name in (
            "pot_id",
            "predicate",
            "subject_key",
            "object_key",
            "valid_at",
            "invalid_at",
            "source_system",
            "source_ref",
            "fact",
            "claim_key",
            "subgraph",
            "truth",
            "description",
            "environment",
            "observed_at",
            "valid_until",
            "mutation_id",
            "graph_contract_version",
            "ontology_version",
            "record_id",
            "evidence_strength",
        )
    },
    "confidence": "DOUBLE",
    "fact_embedding": "DOUBLE[]",
    "source_refs": "STRING[]",
    "retired": "BOOL",
}


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode(value: str) -> Any:
    from io import StringIO

    return strict_json_load(StringIO(value))


def identity(*parts: Any) -> str:
    return encode(parts)


def execute(conn, query, params=None):
    result = conn.execute(query, params or {})
    try:
        if isinstance(result, list):
            raise TypeError("expected exactly one Ladybug query result")
        names = result.get_column_names()
        return [dict(zip(names, row)) for row in result.get_all()]
    finally:
        if isinstance(result, list):
            for item in result:
                item.close()
        else:
            result.close()


def ensure_schema(conn) -> None:
    claim_fields = ", ".join(f"{name} {kind}" for name, kind in CLAIM_COLUMNS.items())
    for ddl in (
        "CREATE NODE TABLE IF NOT EXISTS GraphMetadata(key STRING PRIMARY KEY, version INT64)",
        "CREATE NODE TABLE IF NOT EXISTS Entity(storage_id STRING PRIMARY KEY, pot_id STRING, entity_key STRING, labels STRING[], properties STRING, labels_present BOOL, properties_present BOOL)",
        f"CREATE NODE TABLE IF NOT EXISTS Claim(storage_id STRING PRIMARY KEY, row_order INT64, {claim_fields}, metadata STRING)",
        "CREATE REL TABLE IF NOT EXISTS CLAIM_SUBJECT(FROM Claim TO Entity)",
        "CREATE REL TABLE IF NOT EXISTS CLAIM_OBJECT(FROM Claim TO Entity)",
        "CREATE NODE TABLE IF NOT EXISTS GraphRevision(pot_id STRING PRIMARY KEY, version INT64)",
        "CREATE NODE TABLE IF NOT EXISTS ExecutionReceipt(storage_id STRING PRIMARY KEY, pot_id STRING, mutation_id STRING, payload STRING)",
        "CREATE NODE TABLE IF NOT EXISTS JournalEntry(storage_id STRING PRIMARY KEY, category STRING, pot_id STRING, ordinal INT64, payload STRING)",
    ):
        execute(conn, ddl)
    versions = execute(
        conn, "MATCH (m:GraphMetadata) RETURN m.key AS key, m.version AS version"
    )
    if not versions:
        execute(conn, "CREATE (:GraphMetadata {key: 'schema', version: 1})")
    elif versions != [{"key": "schema", "version": 1}]:
        raise ValueError("unsupported Ladybug graph schema version")


def read_payload(conn) -> dict[str, Any]:
    from potpie_context_engine.core.graph_journal import decode_journal, encode_journal

    entities = execute(
        conn,
        "MATCH (e:Entity) RETURN e.pot_id AS pot_id, e.entity_key AS key, e.labels AS labels, e.properties AS properties, e.labels_present AS labels_present, e.properties_present AS properties_present",
    )
    fields = ", ".join(f"c.{name} AS {name}" for name in CLAIM_COLUMNS)
    claims = execute(
        conn,
        f"MATCH (c:Claim) RETURN {fields}, c.metadata AS metadata ORDER BY c.row_order",
    )
    rows = []
    for claim in claims:
        metadata = decode(claim.pop("metadata"))
        # Preserve absent versus null canonical fields for legacy repair.
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"present", "extra"}
            or not isinstance(metadata["present"], list)
            or not all(isinstance(name, str) for name in metadata["present"])
            or not isinstance(metadata["extra"], dict)
            or set(metadata["extra"]) & CLAIM_COLUMNS.keys()
        ):
            raise ValueError("invalid Ladybug claim metadata")
        present = metadata["present"]
        raw = {name: value for name, value in claim.items() if name in present}
        raw.update(metadata["extra"])
        rows.append(raw)
    revisions = execute(
        conn, "MATCH (r:GraphRevision) RETURN r.pot_id AS pot_id, r.version AS version"
    )
    receipts = execute(
        conn,
        "MATCH (r:ExecutionReceipt) RETURN r.payload AS payload ORDER BY r.storage_id",
    )
    journal = {}
    for entry in execute(
        conn,
        "MATCH (j:JournalEntry) RETURN j.category AS category, j.pot_id AS pot_id, j.ordinal AS ordinal, j.payload AS payload ORDER BY j.ordinal",
    ):
        values = journal.setdefault(entry["category"], {})
        value = decode_journal(decode(entry["payload"]))
        if entry["ordinal"] < 0:
            values[entry["pot_id"]] = value
        else:
            values.setdefault(entry["pot_id"], []).append(value)
    return {
        "schema_version": 1,
        "format_version": "2",
        "rows": rows,
        "labels": [
            [e["pot_id"], e["key"], e["labels"]]
            for e in entities
            if e["labels_present"]
        ],
        "entity_properties": [
            [e["pot_id"], e["key"], decode(e["properties"])]
            for e in entities
            if e["properties_present"]
        ],
        "graph_revisions": {r["pot_id"]: r["version"] for r in revisions},
        "mutation_receipts": [decode(r["payload"]) for r in receipts],
        "graph_journal": encode_journal(journal),
    }


def write_payload(conn, payload: dict[str, Any]) -> None:
    from potpie_context_engine.core.graph_journal import decode_journal, encode_journal

    # The caller owns one native transaction. Even full replacement is invisible
    # until its receipt, graph revision and journal commit together.
    for table in (
        "Claim",
        "Entity",
        "GraphRevision",
        "ExecutionReceipt",
        "JournalEntry",
    ):
        execute(conn, f"MATCH (n:{table}) DETACH DELETE n")
    labels = {(p, k): v for p, k, v in payload["labels"]}
    props = {(p, k): v for p, k, v in payload["entity_properties"]}
    keys = set(labels) | set(props)
    keys.update(
        (r["pot_id"], key)
        for r in payload["rows"]
        for key in (r["subject_key"], r["object_key"])
    )
    for pot_id, key in sorted(keys):
        execute(
            conn,
            "CREATE (:Entity {storage_id: $id, pot_id: $pot, entity_key: $key, labels: $labels, properties: $props, labels_present: $lp, properties_present: $pp})",
            {
                "id": identity(pot_id, key),
                "pot": pot_id,
                "key": key,
                "labels": labels.get((pot_id, key), []),
                "props": encode(props.get((pot_id, key), {})),
                "lp": (pot_id, key) in labels,
                "pp": (pot_id, key) in props,
            },
        )
    for ordinal, row in enumerate(payload["rows"]):
        params = {name: row.get(name) for name in CLAIM_COLUMNS}
        # Ordinal storage identity retains duplicate/missing legacy record IDs;
        # canonical record identity is stored separately and validated by core.
        params["storage_id"] = identity(row["pot_id"], row.get("record_id"), ordinal)
        params["row_order"] = ordinal
        params["metadata"] = encode(
            {
                "present": list(row),
                "extra": {k: v for k, v in row.items() if k not in CLAIM_COLUMNS},
            }
        )
        assignments = ", ".join(f"{name}: ${name}" for name in params)
        execute(conn, f"CREATE (:Claim {{{assignments}}})", params)
        for rel, key in (
            ("CLAIM_SUBJECT", row["subject_key"]),
            ("CLAIM_OBJECT", row["object_key"]),
        ):
            execute(
                conn,
                f"MATCH (c:Claim {{storage_id: $claim}}), (e:Entity {{storage_id: $entity}}) CREATE (c)-[:{rel}]->(e)",
                {
                    "claim": params["storage_id"],
                    "entity": identity(row["pot_id"], key),
                },
            )
    for pot_id, version in payload["graph_revisions"].items():
        execute(
            conn,
            "CREATE (:GraphRevision {pot_id: $pot, version: $version})",
            {"pot": pot_id, "version": version},
        )
    for receipt in payload["mutation_receipts"]:
        execute(
            conn,
            "CREATE (:ExecutionReceipt {storage_id: $id, pot_id: $pot, mutation_id: $mutation, payload: $payload})",
            {
                "id": identity(receipt["pot_id"], receipt["mutation_id"]),
                "pot": receipt["pot_id"],
                "mutation": receipt["mutation_id"],
                "payload": encode(receipt),
            },
        )
    for category, pots in decode_journal(payload["graph_journal"]).items():
        for pot_id, value in pots.items():
            entries = (
                enumerate(value) if isinstance(value, list) and value else [(-1, value)]
            )
            for ordinal, entry in entries:
                execute(
                    conn,
                    "CREATE (:JournalEntry {storage_id: $id, category: $category, pot_id: $pot, ordinal: $ordinal, payload: $payload})",
                    {
                        "id": identity(category, pot_id, ordinal),
                        "category": category,
                        "pot": pot_id,
                        "ordinal": ordinal,
                        "payload": encode(encode_journal(entry)),
                    },
                )
