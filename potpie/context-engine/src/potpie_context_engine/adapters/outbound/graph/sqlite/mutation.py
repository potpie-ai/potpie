"""Transactional SQLite graph mutations with durable apply-once receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_core.graph_contract import evidence_strength_for_truth
from potpie_context_core.graph_entity_summary import merge_entity_display_properties
from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
    ProvenanceRef,
)
from potpie_context_core.ports.graph.mutation import (
    BackendReadiness,
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.reconciliation_validation import (
    validate_reconciliation_plan,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
    MutationExecutionReuseError,
    execution_receipts_from_json,
    execution_receipts_to_json,
    mutation_batch_fingerprint,
)
from potpie_context_engine.adapters.outbound.graph.apply_plan import _build_provenance
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    CONTRACT_EDGE_KEYS,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import card_for_row
from potpie_context_engine.adapters.outbound.graph.sqlite.claim_query import (
    encode_vector,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.connection import (
    SQLiteConnectionFactory,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.schema import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    ensure_schema,
    verify_vector_projection,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_core.ports.claim_query import ClaimRow

_PROFILE = "sqlite"


@dataclass(frozen=True, slots=True)
class _ClaimCard:
    row: ClaimRow
    retrieval_card: str


@dataclass(frozen=True, slots=True)
class _PreparedClaim(_ClaimCard):
    embedding: bytes


@dataclass(slots=True)
class SQLiteMutation:
    """``GraphMutationPort`` over one SQLite canonical/vector transaction."""

    connections: SQLiteConnectionFactory
    embedder: EmbedderPort
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        validated = deepcopy(plan)
        validate_reconciliation_plan(
            validated,
            expected_pot_id,
            definition=self.definition,
            config=reconciliation_config,
        )
        mutation_id = (
            provenance_context.mutation_id
            if provenance_context is not None and provenance_context.mutation_id
            else str(uuid.uuid4())
        )
        fingerprint = mutation_batch_fingerprint(plan)

        # Avoid re-embedding a completed retry. The transaction rechecks this
        # receipt after BEGIN IMMEDIATE to close the concurrent-retry race.
        with self.connections.connect() as connection:
            ensure_schema(connection)
            completed = _lookup_receipt(
                connection,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
                fingerprint=fingerprint,
            )
            if completed.state == MutationExecutionState.completed.value:
                assert completed.result is not None
                return completed.result

        graph_updated_at = datetime.now(timezone.utc)
        provenance = _build_provenance(
            validated,
            pot_id=expected_pot_id,
            mutation_id=mutation_id,
            context=provenance_context,
            graph_updated_at=graph_updated_at,
        )
        cards = tuple(
            self._prepare_claim_card(
                edge,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                provenance=provenance,
                fallback_fact=validated.summary,
                graph_updated_at=graph_updated_at,
            )
            for edge in validated.edge_upserts
        )
        prepared = self._embed_claim_cards(cards)
        if not cards:
            self._probe_embedder()

        with self.connections.connect() as connection:
            ensure_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                completed = _lookup_receipt(
                    connection,
                    expected_pot_id=expected_pot_id,
                    mutation_id=mutation_id,
                    fingerprint=fingerprint,
                )
                if completed.state == MutationExecutionState.completed.value:
                    assert completed.result is not None
                    connection.rollback()
                    return completed.result

                summary = MutationSummary()
                for entity in validated.entity_upserts:
                    self._upsert_entity(
                        connection,
                        pot_id=expected_pot_id,
                        entity=entity,
                        provenance=provenance,
                    )
                    summary.entity_upserts_applied += 1
                for claim in prepared:
                    claim_id = self._upsert_claim(connection, claim)
                    if (
                        claim.row.predicate in self.definition.singleton_predicates
                        and claim.row.evidence_strength == "deterministic"
                    ):
                        self._supersede_singleton_predecessors(
                            connection,
                            row=claim.row,
                            winning_claim_id=claim_id,
                        )
                    summary.edge_upserts_applied += 1
                summary.edge_deletes_applied = self._delete_edges(
                    connection, validated, pot_id=expected_pot_id
                )
                summary.invalidations_applied = self._apply_invalidations(
                    connection, validated, pot_id=expected_pot_id
                )
                result = MutationResult(
                    ok=True,
                    mutation_id=mutation_id,
                    mutation_summary=summary,
                    downgrades=list(validated.ontology_downgrades),
                )
                _write_receipt(
                    connection,
                    pot_id=expected_pot_id,
                    mutation_id=mutation_id,
                    fingerprint=fingerprint,
                    result=result,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    async def apply_async(self, *args: Any, **kwargs: Any) -> MutationResult:
        return await asyncio.to_thread(self.apply, *args, **kwargs)

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        fingerprint = mutation_batch_fingerprint(plan)
        with self.connections.connect() as connection:
            ensure_schema(connection)
            return _lookup_receipt(
                connection,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
                fingerprint=fingerprint,
            )

    def invalidate(
        self,
        *,
        pot_id: str,
        claim_keys: Sequence[str],
        reason: str | None = None,
    ) -> int:
        keys = tuple(dict.fromkeys(str(key) for key in claim_keys if str(key)))
        if not keys:
            return 0
        self._probe_embedder()
        with self.connections.connect() as connection:
            ensure_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = list(
                    connection.execute(
                        """
                        WITH requested(value) AS (
                            SELECT CAST(value AS TEXT) FROM json_each(?)
                        )
                        SELECT claim_id, properties_json
                        FROM claims
                        WHERE pot_id = ?
                          AND invalid_at IS NULL
                          AND (
                              claim_key IN (SELECT value FROM requested)
                              OR subject_key IN (SELECT value FROM requested)
                              OR object_key IN (SELECT value FROM requested)
                          )
                        """,
                        (_json_dump(keys), pot_id),
                    )
                )
                now = datetime.now(timezone.utc).isoformat()
                for row in rows:
                    properties = _json_mapping(row["properties_json"])
                    if reason:
                        properties["invalidation_reason"] = reason
                    connection.execute(
                        """
                        UPDATE claims
                        SET invalid_at = ?, properties_json = ?
                        WHERE claim_id = ?
                        """,
                        (now, _json_dump(properties), int(row["claim_id"])),
                    )
                _delete_vectors(
                    connection, [int(row["claim_id"]) for row in rows]
                )
                connection.commit()
                return len(rows)
            except BaseException:
                connection.rollback()
                raise

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        with self.connections.connect() as connection:
            ensure_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                claim_ids = [
                    int(row["claim_id"])
                    for row in connection.execute(
                        "SELECT claim_id FROM claims WHERE pot_id = ?", (pot_id,)
                    )
                ]
                _delete_vectors(connection, claim_ids)
                connection.execute("DELETE FROM claims WHERE pot_id = ?", (pot_id,))
                connection.execute(
                    "DELETE FROM mutation_receipts WHERE pot_id = ?", (pot_id,)
                )
                connection.execute("DELETE FROM entities WHERE pot_id = ?", (pot_id,))
                connection.commit()
                return {"removed_claims": len(claim_ids)}
            except BaseException:
                connection.rollback()
                raise

    def readiness(self, pot_id: str) -> BackendReadiness:
        del pot_id
        if not self.connections.path.exists():
            return _readiness(
                False,
                f"sqlite graph store has not been provisioned at {self.connections.path}",
            )
        try:
            self._probe_embedder()
            with self.connections.connect() as connection:
                ensure_schema(connection)
                verify_vector_projection(connection)
        except Exception as exc:  # noqa: BLE001 - readiness returns diagnostics.
            return _readiness(False, str(exc))
        return _readiness(
            True,
            "sqlite claim_query + mutation + MiniLM semantic + analytics + "
            "inspection ready; snapshot pending",
        )

    def _probe_embedder(self) -> None:
        encode_vector(self.embedder.embed("potpie sqlite semantic readiness"))

    def _prepare_claim_card(
        self,
        edge: EdgeUpsert,
        *,
        pot_id: str,
        mutation_id: str,
        provenance: ProvenanceRef,
        fallback_fact: str,
        graph_updated_at: datetime,
    ) -> _ClaimCard:
        props = dict(edge.properties)
        truth = _optional_str(props.get("truth"))
        source_ref = _optional_str(props.get("source_ref")) or _stable_source_ref(
            predicate=edge.edge_type,
            from_key=edge.from_entity_key,
            to_key=edge.to_entity_key,
            provenance=provenance,
        )
        source_system = (
            _optional_str(props.get("source_system"))
            or provenance.source_system
            or "agent"
        )
        fact = (
            _optional_str(props.get("fact"))
            or fallback_fact
            or f"{edge.from_entity_key} {edge.edge_type} {edge.to_entity_key}"
        )
        valid_at = (
            _coerce_dt(props.get("valid_at"))
            or provenance.event_occurred_at
            or provenance.valid_from
            or graph_updated_at
        )
        observed_at = _coerce_dt(props.get("observed_at")) or graph_updated_at
        source_refs = _str_tuple(props.get("source_refs"))
        evidence = _evidence_tuple(props.get("evidence"))
        extras = {
            key: value for key, value in props.items() if key not in CONTRACT_EDGE_KEYS
        }
        extras.update(provenance.to_properties())
        extras["provenance_source_event"] = provenance.source_event_id
        row = ClaimRow(
            pot_id=pot_id,
            predicate=edge.edge_type,
            subject_key=edge.from_entity_key,
            object_key=edge.to_entity_key,
            valid_at=valid_at,
            evidence_strength=evidence_strength_for_truth(truth),
            source_system=source_system,
            source_ref=source_ref,
            fact=fact,
            properties=extras,
            claim_key=_optional_str(props.get("claim_key")),
            subgraph=_optional_str(props.get("subgraph")),
            truth=truth,
            confidence=_optional_float(
                props.get("confidence")
                if props.get("confidence") is not None
                else provenance.confidence
            ),
            description=_optional_str(props.get("description")),
            environment=_optional_str(props.get("environment")),
            observed_at=observed_at,
            valid_until=_coerce_dt(props.get("valid_until")),
            mutation_id=mutation_id,
            source_refs=source_refs,
            evidence=evidence,
            graph_contract_version=_optional_str(
                props.get("graph_contract_version")
            ),
            ontology_version=_optional_str(props.get("ontology_version")),
        )
        retrieval_card = card_for_row(row)
        return _ClaimCard(
            row=row,
            retrieval_card=retrieval_card,
        )

    def _embed_claim_cards(
        self, cards: Sequence[_ClaimCard]
    ) -> tuple[_PreparedClaim, ...]:
        if not cards:
            return ()
        vectors = tuple(
            self.embedder.embed_many([card.retrieval_card for card in cards])
        )
        if len(vectors) != len(cards):
            raise RuntimeError(
                "MiniLM returned an unexpected embedding batch size "
                f"({len(vectors)} for {len(cards)} claims)"
            )
        prepared: list[_PreparedClaim] = []
        for card, vector in zip(cards, vectors, strict=True):
            embedding = encode_vector(vector)
            prepared.append(
                _PreparedClaim(
                    row=card.row,
                    retrieval_card=card.retrieval_card,
                    embedding=embedding,
                )
            )
        return tuple(prepared)

    @staticmethod
    def _upsert_entity(
        connection: sqlite3.Connection,
        *,
        pot_id: str,
        entity: EntityUpsert,
        provenance: ProvenanceRef,
    ) -> None:
        existing = connection.execute(
            """
            SELECT properties_json FROM entities
            WHERE pot_id = ? AND entity_key = ?
            """,
            (pot_id, entity.entity_key),
        ).fetchone()
        incoming = dict(entity.properties)
        incoming.update(provenance.to_properties())
        incoming["provenance_source_event"] = provenance.source_event_id
        properties = merge_entity_display_properties(
            incoming,
            existing=_json_mapping(existing["properties_json"])
            if existing is not None
            else {},
            entity_key=entity.entity_key,
        )
        connection.execute(
            """
            INSERT INTO entities(pot_id, entity_key, properties_json)
            VALUES (?, ?, ?)
            ON CONFLICT(pot_id, entity_key)
            DO UPDATE SET properties_json = excluded.properties_json
            """,
            (pot_id, entity.entity_key, _json_dump(properties)),
        )
        connection.execute(
            "DELETE FROM entity_labels WHERE pot_id = ? AND entity_key = ?",
            (pot_id, entity.entity_key),
        )
        connection.executemany(
            """
            INSERT INTO entity_labels(pot_id, entity_key, label)
            VALUES (?, ?, ?)
            """,
            tuple(
                (pot_id, entity.entity_key, label)
                for label in dict.fromkeys(entity.labels)
            ),
        )

    @staticmethod
    def _upsert_claim(
        connection: sqlite3.Connection, prepared: _PreparedClaim
    ) -> int:
        row = prepared.row
        identity_clauses: list[str] = []
        identity_params: list[Any] = [row.pot_id]
        if row.claim_key:
            identity_clauses.append("claim_key = ?")
            identity_params.append(row.claim_key)
        if row.source_ref:
            identity_clauses.append(
                """
                (
                    source_ref = ? AND predicate = ?
                    AND subject_key = ? AND object_key = ?
                )
                """
            )
            identity_params.extend(
                (
                    row.source_ref,
                    row.predicate,
                    row.subject_key,
                    row.object_key,
                )
            )
        existing = None
        if identity_clauses:
            existing = connection.execute(
                f"""
                SELECT claim_id
                FROM claims
                WHERE pot_id = ? AND invalid_at IS NULL
                  AND ({' OR '.join(identity_clauses)})
                ORDER BY claim_id
                LIMIT 1
                """,
                identity_params,
            ).fetchone()

        values = _claim_values(prepared)
        if existing is None:
            cursor = connection.execute(
                f"""
                INSERT INTO claims({_CLAIM_COLUMNS})
                VALUES ({_CLAIM_PLACEHOLDERS})
                """,
                values,
            )
            claim_id = int(cursor.lastrowid)
        else:
            claim_id = int(existing["claim_id"])
            connection.execute(
                f"""
                UPDATE claims
                SET {_CLAIM_UPDATE_ASSIGNMENTS}
                WHERE claim_id = ?
                """,
                (*values, claim_id),
            )
            connection.execute(
                "DELETE FROM claim_vectors WHERE claim_id = ?", (claim_id,)
            )

        connection.execute(
            """
            INSERT INTO claim_vectors(
                claim_id, embedding, pot_id, predicate, subgraph,
                subject_key, object_key, source_system, mutation_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id,
                prepared.embedding,
                row.pot_id,
                row.predicate,
                row.subgraph or "",
                row.subject_key,
                row.object_key,
                row.source_system or "",
                row.mutation_id or "",
            ),
        )
        return claim_id

    @staticmethod
    def _supersede_singleton_predecessors(
        connection: sqlite3.Connection,
        *,
        row: ClaimRow,
        winning_claim_id: int,
    ) -> None:
        losing_ids = [
            int(item["claim_id"])
            for item in connection.execute(
                """
                SELECT claim_id
                FROM claims
                WHERE pot_id = ?
                  AND predicate = ?
                  AND subject_key = ?
                  AND object_key <> ?
                  AND invalid_at IS NULL
                  AND claim_id <> ?
                """,
                (
                    row.pot_id,
                    row.predicate,
                    row.subject_key,
                    row.object_key,
                    winning_claim_id,
                ),
            )
        ]
        if not losing_ids:
            return
        invalid_at = _dt_iso(row.valid_at or datetime.now(timezone.utc))
        connection.execute(
            """
            UPDATE claims
            SET invalid_at = ?
            WHERE claim_id IN (
                SELECT CAST(value AS INTEGER) FROM json_each(?)
            )
            """,
            (invalid_at, _json_dump(losing_ids)),
        )
        _delete_vectors(connection, losing_ids)

    @staticmethod
    def _delete_edges(
        connection: sqlite3.Connection, plan: MutationBatch, *, pot_id: str
    ) -> int:
        count = 0
        for edge in plan.edge_deletes:
            ids = [
                int(row["claim_id"])
                for row in connection.execute(
                    """
                    SELECT claim_id FROM claims
                    WHERE pot_id = ? AND upper(predicate) = ?
                      AND subject_key = ? AND object_key = ?
                    """,
                    (
                        pot_id,
                        edge.edge_type.upper(),
                        edge.from_entity_key,
                        edge.to_entity_key,
                    ),
                )
            ]
            _delete_vectors(connection, ids)
            if ids:
                connection.execute(
                    """
                    DELETE FROM claims
                    WHERE claim_id IN (
                        SELECT CAST(value AS INTEGER) FROM json_each(?)
                    )
                    """,
                    (_json_dump(ids),),
                )
            count += len(ids)
        return count

    @staticmethod
    def _apply_invalidations(
        connection: sqlite3.Connection, plan: MutationBatch, *, pot_id: str
    ) -> int:
        count = 0
        now = datetime.now(timezone.utc)
        for invalidation in plan.invalidations:
            clauses = ["pot_id = ?", "invalid_at IS NULL"]
            params: list[Any] = [pot_id]
            if invalidation.target_edge:
                predicate, subject_key, object_key = invalidation.target_edge
                clauses.extend(
                    [
                        "upper(predicate) = ?",
                        "subject_key = ?",
                        "object_key = ?",
                    ]
                )
                params.extend((predicate.upper(), subject_key, object_key))
            elif invalidation.target_entity_key:
                clauses.append("(subject_key = ? OR object_key = ?)")
                params.extend(
                    (
                        invalidation.target_entity_key,
                        invalidation.target_entity_key,
                    )
                )
            else:
                continue
            rows = list(
                connection.execute(
                    f"""
                    SELECT claim_id, properties_json
                    FROM claims
                    WHERE {' AND '.join(clauses)}
                    """,
                    params,
                )
            )
            invalid_at = _dt_iso(_coerce_dt(invalidation.valid_to) or now)
            for row in rows:
                properties = _json_mapping(row["properties_json"])
                properties["invalidation_reason"] = invalidation.reason
                if invalidation.superseded_by_key:
                    properties["superseded_by_key"] = (
                        invalidation.superseded_by_key
                    )
                connection.execute(
                    """
                    UPDATE claims
                    SET invalid_at = ?, properties_json = ?
                    WHERE claim_id = ?
                    """,
                    (
                        invalid_at,
                        _json_dump(properties),
                        int(row["claim_id"]),
                    ),
                )
            _delete_vectors(connection, [int(row["claim_id"]) for row in rows])
            count += len(rows)
        return count


_CLAIM_COLUMN_NAMES = (
    "pot_id",
    "predicate",
    "subject_key",
    "object_key",
    "claim_key",
    "subgraph",
    "source_system",
    "source_ref",
    "fact",
    "description",
    "retrieval_card",
    "properties_json",
    "source_refs_json",
    "evidence_json",
    "truth",
    "confidence",
    "environment",
    "valid_at",
    "valid_until",
    "invalid_at",
    "observed_at",
    "mutation_id",
    "graph_contract_version",
    "ontology_version",
    "embedding",
    "embedding_model",
    "embedding_dim",
)
_CLAIM_COLUMNS = ", ".join(_CLAIM_COLUMN_NAMES)
_CLAIM_PLACEHOLDERS = ", ".join("?" for _ in _CLAIM_COLUMN_NAMES)
_CLAIM_UPDATE_ASSIGNMENTS = ", ".join(
    f"{column} = ?" for column in _CLAIM_COLUMN_NAMES
)


def _claim_values(prepared: _PreparedClaim) -> tuple[Any, ...]:
    row = prepared.row
    return (
        row.pot_id,
        row.predicate,
        row.subject_key,
        row.object_key,
        row.claim_key,
        row.subgraph,
        row.source_system,
        row.source_ref,
        row.fact,
        row.description,
        prepared.retrieval_card,
        _json_dump(row.properties),
        _json_dump(row.source_refs),
        _json_dump(row.evidence),
        row.truth,
        row.confidence,
        row.environment,
        _dt_iso(row.valid_at),
        _dt_iso(row.valid_until),
        _dt_iso(row.invalid_at),
        _dt_iso(row.observed_at),
        row.mutation_id,
        row.graph_contract_version,
        row.ontology_version,
        prepared.embedding,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )


def _lookup_receipt(
    connection: sqlite3.Connection,
    *,
    expected_pot_id: str,
    mutation_id: str,
    fingerprint: str,
) -> MutationExecutionLookup:
    row = connection.execute(
        """
        SELECT pot_id, batch_fingerprint, result_json
        FROM mutation_receipts
        WHERE mutation_id = ?
        """,
        (mutation_id,),
    ).fetchone()
    if row is None:
        return MutationExecutionLookup(
            state=MutationExecutionState.absent.value,
            mutation_id=mutation_id,
            batch_fingerprint=fingerprint,
        )
    stored_pot_id = str(row["pot_id"])
    stored_fingerprint = str(row["batch_fingerprint"])
    if stored_pot_id != expected_pot_id or stored_fingerprint != fingerprint:
        raise MutationExecutionReuseError(
            f"mutation_id {mutation_id!r} was already used with a different "
            "pot or mutation batch"
        )
    try:
        result_raw = json.loads(str(row["result_json"]))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"durable mutation receipt {mutation_id!r} is corrupt"
        ) from exc
    receipts = execution_receipts_from_json(
        [
            {
                "pot_id": stored_pot_id,
                "mutation_id": mutation_id,
                "batch_fingerprint": stored_fingerprint,
                "result": result_raw,
            }
        ]
    )
    if not receipts:
        raise RuntimeError(f"durable mutation receipt {mutation_id!r} is corrupt")
    return MutationExecutionLookup(
        state=MutationExecutionState.completed.value,
        mutation_id=mutation_id,
        batch_fingerprint=fingerprint,
        result=receipts[0].result,
    )


def _write_receipt(
    connection: sqlite3.Connection,
    *,
    pot_id: str,
    mutation_id: str,
    fingerprint: str,
    result: MutationResult,
) -> None:
    payload = execution_receipts_to_json(
        [
            CompletedMutationExecution(
                pot_id=pot_id,
                mutation_id=mutation_id,
                batch_fingerprint=fingerprint,
                result=result,
            )
        ]
    )[0]["result"]
    connection.execute(
        """
        INSERT INTO mutation_receipts(
            mutation_id, pot_id, batch_fingerprint, result_json
        )
        VALUES (?, ?, ?, ?)
        """,
        (mutation_id, pot_id, fingerprint, _json_dump(payload)),
    )


def _delete_vectors(connection: sqlite3.Connection, claim_ids: Sequence[int]) -> None:
    connection.executemany(
        "DELETE FROM claim_vectors WHERE claim_id = ?",
        tuple((int(claim_id),) for claim_id in claim_ids),
    )


def _readiness(ready: bool, detail: str) -> BackendReadiness:
    return BackendReadiness(
        profile=_PROFILE,
        ready=ready,
        detail=detail,
        capability_ready={
            "mutation": ready,
            "claim_query": ready,
            "semantic": ready,
            "inspection": ready,
            "analytics": ready,
            "snapshot": False,
        },
    )


def _stable_source_ref(
    *,
    predicate: str,
    from_key: str,
    to_key: str,
    provenance: ProvenanceRef,
) -> str:
    if provenance.source_ref:
        return provenance.source_ref
    digest = hashlib.sha256()
    for value in (
        provenance.source_event_id,
        predicate,
        from_key,
        to_key,
    ):
        digest.update(value.encode())
        digest.update(b"\x00")
    return f"event:{provenance.source_event_id}:{digest.hexdigest()[:12]}"


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _evidence_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _coerce_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _dt_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


__all__ = ["SQLiteMutation"]
