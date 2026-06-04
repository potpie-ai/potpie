"""Claim-store primitives — Position B writes/reads.

This is the bare-Neo4j layer that the POC builds everything else on. Same
shape as `pocs/position_b/poc.py`'s primitives, productionized slightly:
- Single shared driver
- Index ensure at startup
- Returns of typed dataclasses where the dict shape isn't load-bearing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncDriver


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    entity_key: str
    labels: tuple[str, ...]
    name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    subject_key: str
    predicate: str
    object_key: str
    source_event_id: str  # the FixtureEvent.event_id this claim was derived from
    source_system: str    # "github" | "linear" | "k8s-scanner" | ...
    evidence_strength: str  # deterministic | attested | inferred | hypothesized
    fact: str
    valid_at: datetime
    environment: str | None = None
    episode_uuid: str | None = None
    confidence: float | None = None  # rarely set; usually derived
    fact_embedding: list[float] | None = None


@dataclass(frozen=True)
class Alias:
    """An alias claim: 'source S asserts that surface_name N refers to canonical entity_key K'."""

    surface_name: str
    canonical_entity_key: str
    source_event_id: str
    source_system: str
    evidence_strength: str
    observed_at: datetime


@dataclass(frozen=True)
class Episode:
    name: str
    body: str
    source_description: str
    reference_time: datetime
    event_id: str


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


async def ensure_indexes(driver: AsyncDriver) -> None:
    queries = [
        # Entity identity
        "CREATE INDEX entity_group_key IF NOT EXISTS "
        "FOR (n:Entity) ON (n.group_id, n.entity_key)",
        # Claim routing
        "CREATE INDEX claim_group_name IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)",
        "CREATE INDEX claim_invalid IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.invalid_at)",
        "CREATE INDEX claim_env IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.environment)",
        # Episodes
        "CREATE INDEX episode_uuid IF NOT EXISTS FOR (e:Episodic) ON (e.uuid)",
        "CREATE INDEX episode_group IF NOT EXISTS FOR (e:Episodic) ON (e.group_id)",
        # Alias surface lookup
        "CREATE INDEX alias_surface IF NOT EXISTS "
        "FOR (a:Alias) ON (a.group_id, a.surface_name_lower)",
        # Vector indexes
        """
        CREATE VECTOR INDEX claim_fact_embeddings IF NOT EXISTS
        FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
        """
        CREATE VECTOR INDEX entity_name_embeddings IF NOT EXISTS
        FOR (n:Entity) ON (n.name_embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
        """
        CREATE VECTOR INDEX episode_body_embeddings IF NOT EXISTS
        FOR (e:Episodic) ON (e.body_embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
    ]
    async with driver.session() as s:
        for q in queries:
            await s.run(q)  # type: ignore[arg-type]


async def cleanup(driver: AsyncDriver, pot: str) -> None:
    async with driver.session() as s:
        await s.run("MATCH (n {group_id: $pot}) DETACH DELETE n", pot=pot)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def write_entity(
    driver: AsyncDriver, pot: str, e: Entity, name_embedding: list[float] | None = None
) -> None:
    label_set = ", ".join(f"n:`{label}`" for label in e.labels)
    cypher = f"""
        MERGE (n:Entity {{group_id: $pot, entity_key: $entity_key}})
        SET {label_set},
            n.name = $name,
            n.uuid = coalesce(n.uuid, randomUUID()),
            n.created_at = coalesce(n.created_at, $now),
            n += $properties
    """
    if name_embedding is not None:
        cypher += ", n.name_embedding = $embedding"
    async with driver.session() as s:
        await s.run(
            cypher,  # type: ignore[arg-type]
            pot=pot,
            entity_key=e.entity_key,
            name=e.name,
            now=now_utc(),
            properties=e.properties,
            embedding=name_embedding,
        )


async def write_episode(
    driver: AsyncDriver, pot: str, ep: Episode, body_embedding: list[float]
) -> str:
    import uuid as _uuid

    episode_uuid = str(_uuid.uuid4())
    async with driver.session() as s:
        await s.run(
            """
            CREATE (e:Episodic {
                uuid: $uuid, group_id: $pot, event_id: $event_id,
                name: $name, body: $body, source_description: $source_description,
                reference_time: $reference_time, created_at: $now,
                body_embedding: $embedding
            })
            """,
            uuid=episode_uuid,
            pot=pot,
            event_id=ep.event_id,
            name=ep.name,
            body=ep.body,
            source_description=ep.source_description,
            reference_time=ep.reference_time,
            now=now_utc(),
            embedding=body_embedding,
        )
    return episode_uuid


async def write_claim(driver: AsyncDriver, pot: str, c: Claim) -> str:
    """Write a claim edge with full bitemporal + provenance properties."""
    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (a:Entity {group_id: $pot, entity_key: $subject_key})
            MATCH (b:Entity {group_id: $pot, entity_key: $object_key})
            MERGE (a)-[r:RELATES_TO {
                group_id: $pot, name: $predicate,
                subject_key: $subject_key, object_key: $object_key,
                source_event_id: $source_event_id
            }]->(b)
            ON CREATE SET r.uuid = randomUUID(), r.created_at = $now,
                          r.invalid_at = null, r.expired_at = null
            SET r.valid_at = $valid_at,
                r.source_system = $source_system,
                r.evidence_strength = $strength,
                r.fact = $fact,
                r.environment = $environment,
                r.episode_uuid = $episode_uuid,
                r.confidence = $confidence,
                r.observed_at = $now,
                r.fact_embedding = $embedding
            RETURN r.uuid AS uuid
            """,
            pot=pot,
            subject_key=c.subject_key,
            object_key=c.object_key,
            predicate=c.predicate,
            source_event_id=c.source_event_id,
            source_system=c.source_system,
            strength=c.evidence_strength,
            fact=c.fact,
            environment=c.environment,
            episode_uuid=c.episode_uuid,
            confidence=c.confidence,
            valid_at=c.valid_at,
            now=now_utc(),
            embedding=c.fact_embedding,
        )
        record = await result.single()
        assert record is not None
        return record["uuid"]


async def write_alias(driver: AsyncDriver, pot: str, alias: Alias) -> None:
    """Write an inspectable alias claim.

    An Alias node records "source S observed surface name N referring to
    canonical entity K". Multiple aliases per canonical entity = corroboration
    of identity. The agent can inspect this table to see WHY two source-level
    names converged on one entity.
    """
    async with driver.session() as s:
        await s.run(
            """
            MATCH (canonical:Entity {group_id: $pot, entity_key: $entity_key})
            MERGE (a:Alias {
                group_id: $pot,
                surface_name_lower: $surface_name_lower,
                canonical_entity_key: $entity_key,
                source_event_id: $source_event_id
            })
            ON CREATE SET a.uuid = randomUUID(), a.created_at = $now
            SET a.surface_name = $surface_name,
                a.source_system = $source_system,
                a.evidence_strength = $strength,
                a.observed_at = $observed_at
            MERGE (a)-[:ALIAS_OF {group_id: $pot}]->(canonical)
            """,
            pot=pot,
            entity_key=alias.canonical_entity_key,
            surface_name=alias.surface_name,
            surface_name_lower=alias.surface_name.strip().lower(),
            source_event_id=alias.source_event_id,
            source_system=alias.source_system,
            strength=alias.evidence_strength,
            observed_at=alias.observed_at,
            now=now_utc(),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def claims_for_subject(
    driver: AsyncDriver,
    pot: str,
    subject_key: str,
    *,
    predicate: str | None = None,
    environment: str | None = None,
    include_invalidated: bool = False,
    as_of: datetime | None = None,
) -> list[dict[str, Any]]:
    where_clauses = ["a.entity_key = $subject_key"]
    if predicate is not None:
        where_clauses.append("r.name = $predicate")
    if environment is not None:
        where_clauses.append("r.environment = $environment")
    if not include_invalidated:
        where_clauses.append("r.invalid_at IS NULL")
    if as_of is not None:
        where_clauses.append("(r.valid_at IS NULL OR r.valid_at <= $as_of)")
        where_clauses.append("(r.invalid_at IS NULL OR r.invalid_at > $as_of)")
    where = " AND ".join(where_clauses)
    cypher = f"""
        MATCH (a:Entity {{group_id: $pot}})-[r:RELATES_TO {{group_id: $pot}}]->(b:Entity)
        WHERE {where}
        RETURN a.entity_key AS subject, r.name AS predicate, b.entity_key AS object,
               r.source_event_id AS source_event_id, r.source_system AS source_system,
               r.evidence_strength AS strength, r.fact AS fact,
               r.valid_at AS valid_at, r.invalid_at AS invalid_at,
               r.environment AS environment, r.confidence AS confidence
        ORDER BY r.valid_at DESC
    """
    async with driver.session() as s:
        r = await s.run(
            cypher,  # type: ignore[arg-type]
            pot=pot,
            subject_key=subject_key,
            predicate=predicate,
            environment=environment,
            as_of=as_of,
        )
        return [dict(rec) async for rec in r]


async def semantic_search_claims(
    driver: AsyncDriver,
    pot: str,
    query_embedding: list[float],
    top_k: int = 10,
    min_score: float = 0.55,
) -> list[dict[str, Any]]:
    cypher = """
        CALL db.index.vector.queryRelationships(
            'claim_fact_embeddings', $top_k, $embedding
        ) YIELD relationship AS r, score
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE r.group_id = $pot AND r.invalid_at IS NULL AND score >= $min_score
        RETURN a.entity_key AS subject, r.name AS predicate, b.entity_key AS object,
               r.source_event_id AS source_event_id, r.source_system AS source_system,
               r.evidence_strength AS strength, r.fact AS fact,
               r.valid_at AS valid_at, r.environment AS environment, score
        ORDER BY score DESC
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=pot, embedding=query_embedding, top_k=top_k, min_score=min_score)
        return [dict(rec) async for rec in r]


async def semantic_search_episodes(
    driver: AsyncDriver,
    pot: str,
    query_embedding: list[float],
    top_k: int = 5,
    min_score: float = 0.55,
) -> list[dict[str, Any]]:
    cypher = """
        CALL db.index.vector.queryNodes(
            'episode_body_embeddings', $top_k, $embedding
        ) YIELD node AS e, score
        WHERE e.group_id = $pot AND score >= $min_score
        RETURN e.uuid AS uuid, e.event_id AS event_id, e.name AS name,
               e.body AS body, e.source_description AS source, score
        ORDER BY score DESC
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=pot, embedding=query_embedding, top_k=top_k, min_score=min_score)
        return [dict(rec) async for rec in r]
