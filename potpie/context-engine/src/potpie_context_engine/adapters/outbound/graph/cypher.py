"""Direct Neo4j Cypher for the Potpie canonical graph.

Rebuild plan P0 (Position B): every claim is one ``:RELATES_TO`` edge
between entities keyed by deterministic ``(group_id, entity_key)``. The
edge carries the predicate (``name``), bitemporal validity
(``valid_at`` / ``invalid_at`` for event time, ``created_at`` /
``expired_at`` for system time), V1.5 claim metadata (``truth``,
``source_refs``, ``claim_key``), and the natural-language fact text the agent
reads + the vector index embeds.

Position B's MERGE key includes ``source_ref`` so two distinct sources
making the same claim land as two corroborating edges, and a re-scan of
the same source updates its edge idempotently in place.

POC reference: ``pocs/position_b/poc.py`` validates this shape end-to-end
including supersession (T6), point-in-time queries (T4), corroboration
(T3), and native-vector semantic search (T9).
"""

from __future__ import annotations

from potpie_context_engine.core.claim_preparation import (
    stable_source_ref as _stable_source_ref,  # noqa: F401 - compatibility export
)

import json
import logging
import re
from datetime import date, datetime, time, timezone
from typing import Any

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_engine.core.entity_canonicalization import coherent_entity_labels
from potpie_context_engine.core.graph_entity_summary import compact_entity_summary
from potpie_context_engine.domain.retrieval_card import build_retrieval_card

logger = logging.getLogger(__name__)


# Property keys on EdgeUpsert.properties that map to first-class edge
# properties on the :RELATES_TO edge. Anything not in this set falls
# through to the catch-all property bag the writer also sets so the
# ontology layer can keep its own keys (e.g. environment, code_scope).
_RESERVED_EDGE_PROPERTY_KEYS: frozenset[str] = frozenset(
    {
        "source_system",
        "source_ref",
        "evidence_strength",
        "fact",
        "fact_embedding",
        "confidence",
        "valid_at",
    }
)

# Properties that mark an edge dead. Re-asserting a claim must clear *all* of
# them, not just ``invalid_at``: a row revived with a stale
# ``superseded_by_object`` still reads live everywhere but explains itself as
# superseded, and ``expired_at`` left set makes system-time queries disagree
# with event-time ones. Written as one Cypher fragment so the assert path and
# the SUPERSEDES path cannot drift apart.
_REVIVE_CLAUSE = """
                    r.invalid_at = null,
                    r.expired_at = null,
                    r.invalidation_reason = null,
                    r.invalidated_by = null,
                    r.superseded_by_object = null,
                    r.supersession_reason = null,
                    r.deleted_by = null"""


def _conditional_revive_clause(*, alias: str, preserve_param: str) -> str:
    """Preserve invalidation lifecycle for metadata-only evidence markers."""
    preserve = f"${preserve_param} AND {alias}.invalid_at IS NOT NULL"
    fields = (
        "invalid_at",
        "expired_at",
        "invalidation_reason",
        "invalidated_by",
        "superseded_by_object",
        "supersession_reason",
        "deleted_by",
    )
    return ",\n                    ".join(
        f"{alias}.{field} = CASE WHEN {preserve} THEN {alias}.{field} ELSE null END"
        for field in fields
    )


_POT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _require_valid_pot_id(pot_id: str) -> None:
    """Fail closed on a malformed partition key at the write boundary.

    Parameter binding already prevents Cypher injection, but an empty /
    whitespace ``pot_id`` reaching a MERGE would act as a *shared*
    partition. Validate here so persistence does not depend solely on the
    upstream policy layer (security review M-3).
    """
    if not isinstance(pot_id, str) or not _POT_ID_RE.match(pot_id):
        raise ValueError(f"invalid pot_id for graph write: {pot_id!r}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(value: Any) -> str | None:
    """Coerce datetime-ish things to ISO strings; pass-through strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _is_valid_predicate(
    name: str, *, definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
) -> bool:
    """Predicate names are interpolated only into ``e.name = ...`` (parameterized);
    still require the canonical-vocab membership to bound surface area.
    """
    if not isinstance(name, str) or not _PREDICATE_RE.match(name):
        return False
    from potpie_context_engine.core.ontology import SYSTEM_EDGE_TYPES

    return name in definition.edge_types or name in SYSTEM_EDGE_TYPES


def _clean_entity_text(value: Any) -> str:
    """One-line-normalize an authored display field; '' when absent."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.strip().split())


def _render_fact(
    *,
    predicate: str,
    from_key: str,
    to_key: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Produce a deterministic, agent-readable text representation of the claim.

    P0 ships a simple template; future phases (P7 ranking, P8 envelope)
    may swap in richer per-predicate renderers without touching writers.
    """
    if extra and isinstance(extra.get("fact"), str) and extra["fact"]:
        return extra["fact"]
    return f"{from_key} {predicate} {to_key}"


def _embedding_props(
    *,
    embedder: Any | None,
    predicate: str,
    from_key: str,
    to_key: str,
    edge_props: dict[str, Any],
) -> dict[str, Any]:
    """Build embedding properties for a claim edge, if an embedder is wired."""
    if embedder is None:
        return {}
    card = build_retrieval_card(
        description=edge_props.get("description")
        if isinstance(edge_props.get("description"), str)
        else None,
        fact=edge_props.get("fact")
        if isinstance(edge_props.get("fact"), str)
        else None,
        subject_key=from_key,
        predicate=predicate,
        object_key=to_key,
        scope=edge_props.get("code_scope")
        if isinstance(edge_props.get("code_scope"), dict)
        else None,
    )
    if not card:
        return {}
    # Embeddings are best-effort enrichment: a model/runtime error must not abort
    # the structural edge write. Degrade to no embedding props and persist anyway.
    try:
        embedding = embedder.embed(card)
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim embedding skipped: %s", exc)
        return {}
    return {
        "fact_embedding": [float(x) for x in embedding],
        "embedding_model": getattr(embedder, "name", "unknown"),
        "embedding_dim": int(getattr(embedder, "dimensions", len(embedding))),
    }


# Temporal types the Neo4j driver maps to native temporal properties — pass
# these through untouched rather than JSON-encoding them.
_NEO4J_TEMPORAL_TYPES = (datetime, date, time)


def _neo4j_safe_value(value: Any) -> Any:
    """Coerce one property value to a Neo4j-storable type.

    Neo4j node/edge properties accept only primitives (``bool``/``int``/
    ``float``/``str``), native temporals, and *homogeneous arrays of
    primitives* — never maps or arrays-of-maps. The V1.5 claim metadata stamps
    structured values (``evidence`` as a list of dicts, ``created_by`` /
    ``code_scope`` as dicts) that the in-memory/embedded backends store
    natively; on Neo4j they are JSON-encoded so the data survives the write
    losslessly (readers that need the structure parse the JSON back). Flat
    arrays (``source_refs``, ``identity_key``) stay native arrays.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, _NEO4J_TEMPORAL_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        # Neo4j 5.x property arrays must be *homogeneous* arrays of primitives —
        # no nulls, no maps, and no mixed element types ([1, "x"] / [True, 1] are
        # rejected at write time). Only a single-primitive-type list survives as a
        # native array; anything else is JSON-encoded so the write does not fail.
        # (bool is a subclass of int, so classify it separately.)
        if all(isinstance(v, (bool, int, float, str)) for v in value):
            kinds = {bool if isinstance(v, bool) else type(v) for v in value}
            if len(kinds) <= 1:  # empty or single-type
                return list(value)
        return json.dumps(list(value), default=str, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    # Unknown type: stringify defensively rather than letting the driver raise.
    return str(value)


def _coerce_props_for_neo4j(props: dict[str, Any]) -> dict[str, Any]:
    """Make every value in a property bag Neo4j-storable (see :func:`_neo4j_safe_value`)."""
    return {k: _neo4j_safe_value(v) for k, v in props.items()}


def _split_edge_properties(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition ``EdgeUpsert.properties`` into (reserved, extras).

    Reserved keys become first-class edge properties on the ``:RELATES_TO``
    edge; extras are merged in alongside so the ontology can layer
    environment / code_scope / lifecycle_status etc. without writer
    changes.
    """
    reserved: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if k in _RESERVED_EDGE_PROPERTY_KEYS:
            reserved[k] = v
        else:
            extras[k] = v
    return reserved, extras


# ----------------------------------------------------------------------
# Entity writes
# ----------------------------------------------------------------------


async def upsert_entities_async(
    driver: Any,
    pot_id: str,
    items: list[EntityUpsert],
    provenance: ProvenanceRef,
    *,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
) -> int:
    _require_valid_pot_id(pot_id)
    if not items:
        return 0
    count = 0
    prov_props = provenance.to_properties()
    async with driver.session() as session:
        for item in items:
            if item.preserve_labels:
                result = await session.run(
                    "MATCH (e:Entity {group_id:$gid,entity_key:$key}) SET e += $props RETURN count(e) AS cnt",
                    gid=pot_id,
                    key=item.entity_key,
                    props=_coerce_props_for_neo4j({**item.properties, **prov_props}),
                )
                record = await result.single()
                await result.consume()
                if record is None or int(record["cnt"]) != 1:
                    from potpie_context_engine.core.claim_lifecycle import (
                        ClaimLifecycleConflict,
                    )

                    raise ClaimLifecycleConflict(
                        "entity projection target is missing or ambiguous"
                    )
                count += 1
                continue
            labels = coherent_entity_labels(
                item.entity_key,
                item.labels,
                entity_types=definition.entity_types,
            )
            props = dict(item.properties)
            # Authored display/retrieval fields overwrite; key-derived
            # fallbacks only fill nodes that have no value yet, so a bare
            # re-reference (key + type only) never clobbers an authored
            # summary/description already stored on the node.
            authored_name = _clean_entity_text(props.pop("name", None))
            authored_summary = compact_entity_summary(
                props.pop("summary", None),
                props.get("description"),
                props.get("title"),
                authored_name,
            )
            authored_description = (
                _clean_entity_text(props.pop("description", None)) or authored_summary
            )
            props["group_id"] = pot_id
            props["provenance_source_event"] = provenance.source_event_id
            props.update(prov_props)
            await session.run(
                "MERGE (e:Entity {group_id: $gid, entity_key: $key}) "
                "ON CREATE SET e.uuid = randomUUID(), e.created_at = timestamp() "
                "SET e += $props "
                "SET e.name = CASE WHEN $a_name <> '' THEN $a_name "
                "WHEN coalesce(e.name, '') = '' THEN $key ELSE e.name END, "
                "e.summary = CASE WHEN $a_summary <> '' THEN $a_summary "
                "WHEN coalesce(e.summary, '') = '' THEN $key ELSE e.summary END, "
                "e.description = CASE WHEN $a_description <> '' THEN $a_description "
                "WHEN coalesce(e.description, '') = '' THEN $key ELSE e.description END",
                gid=pot_id,
                key=item.entity_key,
                props=_coerce_props_for_neo4j(props),
                a_name=authored_name,
                a_summary=authored_summary,
                a_description=authored_description,
            )
            valid_entity_types = {
                label for label in definition.entity_types if _LABEL_RE.fullmatch(label)
            }
            wanted_labels = {label for label in labels if label in valid_entity_types}
            stale_labels = sorted(valid_entity_types - wanted_labels)
            if stale_labels:
                remove_clause = ":".join(stale_labels)
                await session.run(
                    f"MATCH (e:Entity {{group_id: $gid, entity_key: $key}}) REMOVE e:{remove_clause}",  # pyright: ignore[reportArgumentType]
                    gid=pot_id,
                    key=item.entity_key,
                )
            for lbl in sorted(wanted_labels):
                await session.run(
                    f"MATCH (e:Entity {{group_id: $gid, entity_key: $key}}) SET e:{lbl}",  # pyright: ignore[reportArgumentType]
                    gid=pot_id,
                    key=item.entity_key,
                )
            count += 1
    return count


# ----------------------------------------------------------------------
# Edge writes — Position B :RELATES_TO shape
# ----------------------------------------------------------------------


async def upsert_edges_async(
    driver: Any,
    pot_id: str,
    items: list[EdgeUpsert],
    provenance: ProvenanceRef,
    *,
    embedder: Any | None = None,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
) -> int:
    """Persist physical incarnations using the canonical lifecycle policy."""

    from .cypher_claim_writes import write_claim

    _require_valid_pot_id(pot_id)
    count = 0
    async with driver.session() as session:
        for item in items:
            if not _is_valid_predicate(item.edge_type, definition=definition):
                continue
            await write_claim(
                session,
                pot_id=pot_id,
                edge=item,
                provenance=provenance,
                definition=definition,
                embedder=embedder,
            )
            count += 1
    return count


# ----------------------------------------------------------------------
# Edge deletes — soft-delete via invalid_at, per P0 audit-preserving model
# ----------------------------------------------------------------------


async def delete_edges_async(
    driver: Any,
    pot_id: str,
    items: list[EdgeDelete],
    provenance: ProvenanceRef,
    *,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
) -> int:
    from .cypher_claim_writes import close_claims
    _require_valid_pot_id(pot_id)
    anchor = provenance.graph_updated_at or _utc_now()
    count = 0
    async with driver.session() as session:
        for item in items:
            count += await close_claims(
                session,
                pot_id=pot_id,
                closures=item.closures,
                properties={
                    "expired_at": anchor.isoformat(),
                    "deleted_by": provenance.source_event_id,
                    "last_mutation_id": provenance.mutation_id,
                },
            )
    return count


async def apply_invalidations_async(
    driver: Any,
    pot_id: str,
    items: list[InvalidationOp],
    provenance: ProvenanceRef,
    *,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
) -> int:
    """Apply frozen claim intervals. Derived effects are ordinary batch writes."""
    from .cypher_claim_writes import close_claims
    _require_valid_pot_id(pot_id)
    anchor = provenance.graph_updated_at or _utc_now()
    count = 0
    async with driver.session() as session:
        for item in items:
            count += await close_claims(
                session,
                pot_id=pot_id,
                closures=item.closures,
                properties={
                    "expired_at": anchor.isoformat(),
                    "invalidation_reason": item.reason,
                    "invalidated_by": provenance.source_event_id,
                    "last_mutation_id": provenance.mutation_id,
                },
            )
    return count


# ----------------------------------------------------------------------
# Index management (idempotent; called from bootstrap)
# ----------------------------------------------------------------------


async def ensure_canonical_indexes(driver: Any, *, embedding_dim: int = 1536) -> None:
    """Create the indexes the Position B traversal patterns rely on.

    - ``entity_group_key``: pot-scoped entity lookup by deterministic key.
    - ``claim_group_name``: ``(group_id, name)`` for predicate-filtered
      traversal (the hot path for blast-radius queries).
    - ``claim_temporal``: ``(group_id, invalid_at)`` so live-set
      filtering is index-backed.
    - ``claim_fact_embeddings``: native Neo4j 5.x relationship vector
      index on ``r.fact_embedding``. Idempotent — re-runs are no-ops.

    Embedding dimension defaults to 1536 (OpenAI text-embedding-3-small);
    callers may pass an alternative via ``embedding_dim``.
    """
    async with driver.session() as session:
        await session.run(
            "CREATE INDEX entity_group_key IF NOT EXISTS "
            "FOR (n:Entity) ON (n.group_id, n.entity_key)"
        )
        await session.run(
            "CREATE INDEX claim_group_name IF NOT EXISTS "
            "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)"
        )
        await session.run(
            "CREATE INDEX claim_temporal IF NOT EXISTS "
            "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.invalid_at)"
        )
        # Event-time index for timeline window/recency scans
        # (``r.valid_at >= $va_after`` is the hot path for activity readers).
        await session.run(
            "CREATE INDEX claim_valid_at IF NOT EXISTS "
            "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name, r.valid_at)"
        )
        # Vector index for semantic search over claim facts (UC4).
        # Wrapped because some Neo4j editions / community versions don't
        # support relationship vector indexes; the rest of the system
        # still works without it (read paths fall back to non-vector).
        try:
            await session.run(
                """
                CREATE VECTOR INDEX claim_fact_embeddings IF NOT EXISTS
                FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: $embedding_dim,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                embedding_dim=int(embedding_dim),
            )
        except Exception as exc:
            logger.warning(
                "claim_fact_embeddings vector index unavailable: %s "
                "(semantic search will degrade to exact-text fallback)",
                exc,
            )
