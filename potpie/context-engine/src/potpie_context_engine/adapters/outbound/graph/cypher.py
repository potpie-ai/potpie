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

import hashlib
import json
import logging
import re
from datetime import date, datetime, time, timezone
from typing import Any

from potpie_context_core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_core.graph_entity_summary import compact_entity_summary
from potpie_context_core.graph_contract import evidence_strength_for_truth
from potpie_context_core.ontology import (
    CANONICAL_EDGE_TYPES,
    ENTITY_TYPES,
    canonical_entity_labels,
    is_canonical_entity_label,
)
from potpie_context_engine.domain.retrieval_card import build_retrieval_card
from potpie_context_core.singleton_predicates import is_singleton_predicate

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

_POT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


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


def _is_valid_predicate(name: str) -> bool:
    """Predicate names are interpolated only into ``e.name = ...`` (parameterized);
    still require the canonical-vocab membership to bound surface area.
    """
    if not isinstance(name, str) or not _PREDICATE_RE.match(name):
        return False
    return name in CANONICAL_EDGE_TYPES


def _clean_entity_text(value: Any) -> str:
    """One-line-normalize an authored display field; '' when absent."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.strip().split())


def _stable_source_ref(
    *,
    predicate: str,
    from_key: str,
    to_key: str,
    provenance: ProvenanceRef,
) -> str:
    """Derive a deterministic ``source_ref`` when the caller did not supply one.

    The MERGE key includes ``source_ref`` so corroborating writes from
    different sources stay distinct; without it every claim from the
    same predicate triple would collide and look like a single source.
    When the caller provides ``source_ref`` explicitly via
    ``EdgeUpsert.properties``, we honor that.

    The fallback is ``provenance.source_ref`` if set, else a hash over the
    source_event_id + the edge triple. This stays stable across re-scans
    of the same event and stays unique across distinct events.
    """
    if provenance.source_ref:
        return provenance.source_ref
    digest = hashlib.sha256()
    digest.update(provenance.source_event_id.encode())
    digest.update(b"\x00")
    digest.update(predicate.encode())
    digest.update(b"\x00")
    digest.update(from_key.encode())
    digest.update(b"\x00")
    digest.update(to_key.encode())
    return f"event:{provenance.source_event_id}:{digest.hexdigest()[:12]}"


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
) -> int:
    _require_valid_pot_id(pot_id)
    if not items:
        return 0
    count = 0
    prov_props = provenance.to_properties()
    async with driver.session() as session:
        for item in items:
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
            wanted_labels = set(canonical_entity_labels(item.labels))
            if wanted_labels:
                for stale in sorted(set(ENTITY_TYPES) - wanted_labels):
                    await session.run(
                        f"MATCH (e:Entity {{group_id: $gid, entity_key: $key}}) REMOVE e:{stale}",  # pyright: ignore[reportArgumentType]
                        gid=pot_id,
                        key=item.entity_key,
                    )
            for lbl in item.labels:
                if lbl == "Entity":
                    continue
                if not is_canonical_entity_label(lbl) or lbl not in ENTITY_TYPES:
                    continue
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
) -> int:
    """Upsert every edge as a ``:RELATES_TO {name: <predicate>, ...}`` claim.

    Each call produces a single MERGE keyed on
    ``(group_id, name, subject_key, object_key, source_ref)`` so re-scans
    update in place and corroborating writes from distinct sources land
    as distinct edges.

    Properties set on the edge:

    - **Identity / MERGE key:** ``group_id``, ``name``, ``subject_key``,
      ``object_key``, ``source_ref``.
    - **System time:** ``uuid``, ``created_at`` (ON CREATE only),
      ``expired_at`` (set to null ON CREATE).
    - **Event time:** ``valid_at`` (per-claim, from provenance or
      explicit), ``invalid_at`` (null until supersession).
    - **Provenance:** ``source_system``, ``evidence_strength``, ``fact``,
      ``confidence`` (optional), ``observed_at`` (write time).
    - **Ontology extras:** whatever else the caller put on
      ``EdgeUpsert.properties`` (e.g. ``environment``, ``code_scope``).
    """
    _require_valid_pot_id(pot_id)
    if not items:
        return 0
    now = _utc_now()
    count = 0
    async with driver.session() as session:
        for item in items:
            predicate = item.edge_type
            if not _is_valid_predicate(predicate):
                logger.warning(
                    "skipping edge with unknown predicate name=%r "
                    "(must be uppercase identifier in CANONICAL_EDGE_TYPES)",
                    predicate,
                )
                continue

            reserved, extras = _split_edge_properties(item.properties)

            source_ref = _stable_source_ref(
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                provenance=provenance,
            )
            if isinstance(reserved.get("source_ref"), str) and reserved["source_ref"]:
                source_ref = reserved["source_ref"]

            valid_at = (
                _iso(reserved.get("valid_at"))
                or _iso(provenance.event_occurred_at)
                or _iso(provenance.valid_from)
                or now.isoformat()
            )

            source_system = (
                reserved.get("source_system") or provenance.source_system or "agent"
            )

            truth = (
                extras.get("truth") if isinstance(extras.get("truth"), str) else None
            )
            evidence_strength = evidence_strength_for_truth(truth)

            fact = _render_fact(
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                extra=reserved,
            )

            confidence_raw = reserved.get("confidence")
            if confidence_raw is None:
                confidence_raw = provenance.confidence
            confidence = float(confidence_raw) if confidence_raw is not None else None

            edge_props: dict[str, Any] = {
                "valid_at": valid_at,
                "source_system": source_system,
                "evidence_strength": evidence_strength,
                "fact": fact,
                "observed_at": now.isoformat(),
            }
            if confidence is not None:
                edge_props["confidence"] = confidence
            # Carry the ontology-specific extras alongside POC fields.
            for k, v in extras.items():
                edge_props[k] = v
            edge_props.update(
                _embedding_props(
                    embedder=embedder,
                    predicate=predicate,
                    from_key=item.from_entity_key,
                    to_key=item.to_entity_key,
                    edge_props=edge_props,
                )
            )
            # Stamp a compact provenance pointer; full provenance is on
            # the source event that source_ref references.
            edge_props["provenance_source_event"] = provenance.source_event_id
            if provenance.mutation_id:
                edge_props["mutation_id"] = provenance.mutation_id

            await session.run(
                """
                MATCH (a:Entity {group_id: $gid, entity_key: $from_key})
                MATCH (b:Entity {group_id: $gid, entity_key: $to_key})
                MERGE (a)-[r:RELATES_TO {
                    group_id: $gid,
                    name: $predicate,
                    subject_key: $from_key,
                    object_key: $to_key,
                    source_ref: $source_ref
                }]->(b)
                ON CREATE SET
                    r.uuid = randomUUID(),
                    r.created_at = $now,
                    r.expired_at = null,
                    r.invalid_at = null
                SET r += $props
                """,
                gid=pot_id,
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                source_ref=source_ref,
                now=now.isoformat(),
                props=_coerce_props_for_neo4j(edge_props),
            )
            # F3 deterministic supersession: when a singleton-predicate
            # claim from a deterministic-strength source lands, invalidate
            # all prior live claims for the same subject+predicate that
            # point at a *different* object. Multi-source corroboration on
            # the same object is preserved (different source_ref, same
            # object → no supersession trigger).
            if (
                is_singleton_predicate(predicate)
                and evidence_strength == "deterministic"
            ):
                await _supersede_singleton_predecessors(
                    session,
                    pot_id=pot_id,
                    predicate=predicate,
                    from_key=item.from_entity_key,
                    winning_to_key=item.to_entity_key,
                    valid_at=valid_at,
                    now=now,
                )
            count += 1
    return count


async def _supersede_singleton_predecessors(
    session: Any,
    *,
    pot_id: str,
    predicate: str,
    from_key: str,
    winning_to_key: str,
    valid_at: str,
    now: datetime,
) -> int:
    """Stamp ``invalid_at`` on prior live singleton claims that disagree.

    Cypher matches every live ``:RELATES_TO`` edge from ``from_key`` with
    predicate ``name`` whose object is *not* the new winner; stamps
    them with ``invalid_at = $new_valid_at`` (event time of the new
    claim) and ``expired_at = $now`` (system time of the supersession).

    Note: this does NOT remove the prior edges — point-in-time queries
    still see them as live for ``as_of < invalid_at``. The bitemporal
    history stays intact.
    """
    result = await session.run(
        """
        MATCH (a:Entity {group_id: $gid, entity_key: $from_key})
              -[r:RELATES_TO {
                   group_id: $gid,
                   name: $predicate,
                   subject_key: $from_key
              }]->(b:Entity)
        WHERE b.entity_key <> $winning_to_key
          AND r.invalid_at IS NULL
        SET r.invalid_at = $new_valid_at,
            r.expired_at = $now,
            r.superseded_by_object = $winning_to_key,
            r.supersession_reason = 'singleton_predicate'
        RETURN count(r) AS cnt
        """,
        gid=pot_id,
        predicate=predicate,
        from_key=from_key,
        winning_to_key=winning_to_key,
        new_valid_at=valid_at,
        now=now.isoformat(),
    )
    rec = await result.single()
    await result.consume()
    return int(rec["cnt"]) if rec is not None else 0


# ----------------------------------------------------------------------
# Edge deletes — soft-delete via invalid_at, per P0 audit-preserving model
# ----------------------------------------------------------------------


async def delete_edges_async(
    driver: Any,
    pot_id: str,
    items: list[EdgeDelete],
    provenance: ProvenanceRef,
) -> int:
    """Logically delete a claim by stamping ``invalid_at`` + ``expired_at``.

    P0 changes the semantics: claims are append-only, point-in-time
    queries depend on the bitemporal predicate; an actual DELETE would
    destroy history. The lifecycle machinery (auto-supersede + conflict
    detection) operates on the same ``invalid_at`` property, so this
    matches their model.

    Note: this matches *every* live claim for the (subject, predicate,
    object) triple across all source_refs — that's the closest equivalent
    to the old hard-delete semantics. Single-source invalidation should
    go through ``apply_invalidations_async`` with a target_edge that
    carries the source_ref.
    """
    _require_valid_pot_id(pot_id)
    if not items:
        return 0
    now = _utc_now()
    count = 0
    async with driver.session() as session:
        for item in items:
            predicate = item.edge_type
            if not _is_valid_predicate(predicate):
                continue
            res = await session.run(
                """
                MATCH (a:Entity {group_id: $gid, entity_key: $from_key})
                       -[r:RELATES_TO {
                            group_id: $gid,
                            name: $predicate,
                            subject_key: $from_key,
                            object_key: $to_key
                       }]->
                       (b:Entity {group_id: $gid, entity_key: $to_key})
                WHERE r.invalid_at IS NULL
                SET r.invalid_at = $now,
                    r.expired_at = $now,
                    r.deleted_by = $deleted_by
                RETURN count(r) AS cnt
                """,
                gid=pot_id,
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                now=now.isoformat(),
                deleted_by=provenance.source_event_id,
            )
            rec = await res.single()
            await res.consume()
            count += int(rec["cnt"]) if rec is not None else 0
    return count


# ----------------------------------------------------------------------
# Invalidations — entity and edge with bitemporal stamps + SUPERSEDES
# ----------------------------------------------------------------------


async def apply_invalidations_async(
    driver: Any,
    pot_id: str,
    items: list[InvalidationOp],
    provenance: ProvenanceRef,
) -> int:
    """Stamp ``invalid_at`` on entities/edges, optionally with a SUPERSEDES claim.

    Unifies on the bitemporal naming (``invalid_at`` / ``expired_at``).
    When ``superseded_by_key`` is set, also writes a ``:RELATES_TO
    {name: 'SUPERSEDES'}`` claim from the new entity to the invalidated
    one so the supersession relationship is itself a queryable claim.
    """
    _require_valid_pot_id(pot_id)
    if not items:
        return 0
    now = _utc_now()
    count = 0
    async with driver.session() as session:
        for item in items:
            valid_to = _iso(item.valid_to) or now.isoformat()
            invalidation_props: dict[str, Any] = {
                "invalid_at": valid_to,
                "expired_at": now.isoformat(),
                "invalidation_reason": item.reason,
                "invalidated_by": provenance.source_event_id,
            }

            if item.target_entity_key:
                res = await session.run(
                    "MATCH (e:Entity {group_id: $gid, entity_key: $key}) "
                    "SET e += $props "
                    "RETURN count(e) AS cnt",
                    gid=pot_id,
                    key=item.target_entity_key,
                    props=invalidation_props,
                )
                rec = await res.single()
                await res.consume()
                matched = int(rec["cnt"]) if rec is not None else 0
                if matched and item.superseded_by_key:
                    await _write_supersedes_claim(
                        session,
                        pot_id=pot_id,
                        new_key=item.superseded_by_key,
                        old_key=item.target_entity_key,
                        reason=item.reason,
                        now=now,
                        provenance=provenance,
                    )
                count += matched
            elif item.target_edge:
                edge_type, from_key, to_key = item.target_edge
                if not _is_valid_predicate(edge_type):
                    continue
                res = await session.run(
                    """
                    MATCH (a:Entity {group_id: $gid, entity_key: $from_key})
                           -[r:RELATES_TO {
                                group_id: $gid,
                                name: $predicate,
                                subject_key: $from_key,
                                object_key: $to_key
                           }]->
                           (b:Entity {group_id: $gid, entity_key: $to_key})
                    WHERE r.invalid_at IS NULL
                    SET r += $props
                    RETURN count(r) AS cnt
                    """,
                    gid=pot_id,
                    predicate=edge_type,
                    from_key=from_key,
                    to_key=to_key,
                    props=invalidation_props,
                )
                rec = await res.single()
                await res.consume()
                matched = int(rec["cnt"]) if rec is not None else 0
                if matched and item.superseded_by_key:
                    await _write_supersedes_claim(
                        session,
                        pot_id=pot_id,
                        new_key=item.superseded_by_key,
                        old_key=to_key,
                        reason=item.reason,
                        now=now,
                        provenance=provenance,
                    )
                count += matched
    return count


async def _write_supersedes_claim(
    session: Any,
    *,
    pot_id: str,
    new_key: str,
    old_key: str,
    reason: str,
    now: datetime,
    provenance: ProvenanceRef,
) -> None:
    """Write ``SUPERSEDES`` as a normal :RELATES_TO claim so it follows the same
    audit + bitemporal rules as every other edge.
    """
    source_ref = _stable_source_ref(
        predicate="SUPERSEDES",
        from_key=new_key,
        to_key=old_key,
        provenance=provenance,
    )
    fact = f"{new_key} supersedes {old_key}: {reason}"
    await session.run(
        """
        MATCH (a:Entity {group_id: $gid, entity_key: $new_key})
        MATCH (b:Entity {group_id: $gid, entity_key: $old_key})
        MERGE (a)-[r:RELATES_TO {
            group_id: $gid,
            name: 'SUPERSEDES',
            subject_key: $new_key,
            object_key: $old_key,
            source_ref: $source_ref
        }]->(b)
        ON CREATE SET
            r.uuid = randomUUID(),
            r.created_at = $now_iso,
            r.expired_at = null,
            r.invalid_at = null
        SET
            r.valid_at = $now_iso,
            r.source_system = $source_system,
            r.evidence_strength = 'deterministic',
            r.fact = $fact,
            r.observed_at = $now_iso,
            r.reason = $reason
        """,
        gid=pot_id,
        new_key=new_key,
        old_key=old_key,
        source_ref=source_ref,
        now_iso=now.isoformat(),
        source_system=provenance.source_system or "agent",
        fact=fact,
        reason=reason,
    )


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
