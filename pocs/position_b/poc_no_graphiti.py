#!/usr/bin/env python
"""
Position B + Graphiti-free POC.

Builds on poc.py and validates we can run the full claim-store substrate
(write episodes, write claims, supersede, detect conflicts, scale, search)
**without importing graphiti_core anywhere**. The intent is to size the
Graphiti-removal decision honestly.

Tests
-----
T10  episode persistence without Graphiti     — write Episodic + embedding ourselves
T11  multi-label entity                       — SET n:Service:Activity works
T12  environment-scoped claim                 — env-filtered traversal
T13  re-scan idempotency                      — same source_ref → one edge, in-place update
T14  episode → claim provenance               — navigate claim back to its source episode
T15  temporal auto-supersede (full mechanic)  — time-ordered → supersession; equal-time → conflict
T16  family-conflict QualityIssue creation    — equal-strength contradiction surfaces a conflict
T17  scale test                               — 1000 entities, 5000 claims, traversal latency
T18  episode semantic search                  — find episode by query similarity, navigate to derived claims
T19  no-graphiti-import assertion             — confirm graphiti_core is never imported in this script

Run:
    cd /Users/nandan/Desktop/Dev/potpie
    .venv/bin/python pocs/position_b/poc_no_graphiti.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

POT = "pot:position-b-deep-poc"

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def header(t: str) -> None:
    print(f"\n{'━' * 76}\n  {t}\n{'━' * 76}")


def step(t: str) -> None:
    print(f"\n→ {t}")


def ok(msg: str, detail: str = "") -> None:
    print(f"  {OK} {msg}{(' — ' + detail) if detail else ''}")


def fail(msg: str, detail: str = "") -> None:
    print(f"  {FAIL} {msg}{(' — ' + detail) if detail else ''}")
    sys.exit(1)


def assert_eq(actual: Any, expected: Any, what: str) -> None:
    if actual == expected:
        ok(what, str(actual))
    else:
        fail(what, f"expected {expected!r}, got {actual!r}")


def assert_true(cond: bool, what: str, detail: str = "") -> None:
    (ok if cond else fail)(what, detail)


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def new_uuid() -> str:
    return str(_uuid.uuid4())


# ===========================================================================
# Substrate: write_episode, write_entity, write_claim — NO graphiti imports
# ===========================================================================


@dataclass(frozen=True)
class Episode:
    """The raw audit record of an ingested event. Replaces g.add_episode()."""

    name: str
    body: str
    source_description: str
    reference_time: datetime


async def write_episode(driver: AsyncDriver, pot: str, ep: Episode) -> str:
    """Episode persistence, 30 LoC, no Graphiti.

    This is the *entire* replacement for graphiti.add_episode() in the
    Position-B-and-no-Graphiti world. It writes an :Episodic node with:
      - a UUID
      - the raw body (audit trail)
      - source_description + reference_time (when did it happen)
      - created_at (when we observed it)
      - body_embedding (for semantic episode search later)

    Returns the episode uuid.
    """
    embedding = await embed_text(ep.body)
    episode_uuid = new_uuid()
    cypher = """
        CREATE (e:Episodic {
            uuid: $uuid,
            group_id: $pot,
            name: $name,
            body: $body,
            source_description: $source_description,
            reference_time: $reference_time,
            created_at: $now,
            body_embedding: $embedding
        })
        RETURN e.uuid AS uuid
    """
    async with driver.session() as s:
        r = await s.run(
            cypher,
            uuid=episode_uuid,
            pot=pot,
            name=ep.name,
            body=ep.body,
            source_description=ep.source_description,
            reference_time=ep.reference_time,
            now=now(),
            embedding=embedding,
        )
        record = await r.single()
        assert record is not None
        return record["uuid"]


@dataclass(frozen=True)
class Entity:
    entity_key: str
    labels: tuple[str, ...]  # multi-label support: ("Service",) or ("Service", "Activity")
    name: str
    properties: dict[str, Any] | None = None


async def write_entity(driver: AsyncDriver, pot: str, e: Entity) -> None:
    label_clause = ", ".join(f"n:`{label}`" for label in e.labels)
    cypher = f"""
        MERGE (n:Entity {{group_id: $pot, entity_key: $entity_key}})
        SET {label_clause},
            n.name = $name,
            n.uuid = coalesce(n.uuid, randomUUID()),
            n.created_at = coalesce(n.created_at, $now)
    """
    if e.properties:
        cypher += ", n += $properties"
    async with driver.session() as s:
        await s.run(
            cypher,
            pot=pot,
            entity_key=e.entity_key,
            name=e.name,
            now=now(),
            properties=e.properties or {},
        )


@dataclass(frozen=True)
class Claim:
    subject_key: str
    predicate: str
    object_key: str
    source_system: str
    source_ref: str
    evidence_strength: str
    fact: str
    valid_at: datetime
    environment: str | None = None  # for env-scoped claims
    episode_uuid: str | None = None  # link back to the source episode
    confidence: float | None = None


async def write_claim(driver: AsyncDriver, pot: str, c: Claim) -> str:
    cypher = """
        MATCH (a:Entity {group_id: $pot, entity_key: $subject_key})
        MATCH (b:Entity {group_id: $pot, entity_key: $object_key})
        MERGE (a)-[r:RELATES_TO {
            group_id: $pot,
            name: $predicate,
            subject_key: $subject_key,
            object_key: $object_key,
            source_ref: $source_ref
        }]->(b)
        ON CREATE SET
            r.uuid = randomUUID(),
            r.created_at = $now,
            r.expired_at = null,
            r.invalid_at = null
        SET
            r.valid_at = $valid_at,
            r.source_system = $source_system,
            r.evidence_strength = $strength,
            r.fact = $fact,
            r.confidence = $confidence,
            r.observed_at = $now,
            r.environment = $environment,
            r.episode_uuid = $episode_uuid
        RETURN r.uuid AS uuid
    """
    async with driver.session() as s:
        r = await s.run(
            cypher,
            pot=pot,
            subject_key=c.subject_key,
            object_key=c.object_key,
            predicate=c.predicate,
            source_ref=c.source_ref,
            source_system=c.source_system,
            strength=c.evidence_strength,
            fact=c.fact,
            confidence=c.confidence,
            valid_at=c.valid_at,
            now=now(),
            environment=c.environment,
            episode_uuid=c.episode_uuid,
        )
        record = await r.single()
        assert record is not None
        return record["uuid"]


# ===========================================================================
# Temporal auto-supersession (adapted from temporal_supersede.py)
# ===========================================================================


async def apply_predicate_family_supersede(
    driver: AsyncDriver, pot: str, *, family_filter: set[str] | None = None
) -> dict[str, Any]:
    """Walk live claims grouped by (predicate, subject); within each group,
    if multiple distinct objects exist, the latest by valid_at wins; older
    distinct-object claims get invalid_at stamped.

    Equal-time same-strength contradictions are LEFT live for the conflict
    detector to surface (T16).

    Mirrors temporal_supersede.py's logic but operates on Potpie's
    canonical edges with our entity_keys instead of Graphiti UUIDs.
    """
    family_predicates = family_filter or {
        "DEPENDS_ON",
        "STORED_IN",
        "DEPLOYED_TO",
        "OWNED_BY",
    }
    invalidated = 0
    audit: list[dict[str, Any]] = []

    cypher_live = """
        MATCH (a:Entity)-[r:RELATES_TO {group_id: $pot}]->(b:Entity)
        WHERE r.invalid_at IS NULL AND r.name IN $predicates
        RETURN r.uuid AS uuid, r.name AS predicate,
               a.entity_key AS subject, b.entity_key AS object,
               r.valid_at AS valid_at, r.evidence_strength AS strength
    """
    async with driver.session() as s:
        rs = await s.run(cypher_live, pot=pot, predicates=list(family_predicates))
        rows = [dict(rec) async for rec in rs]

    # Bucket by (predicate, subject)
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["predicate"], r["subject"])].append(r)

    strength_rank = {
        "deterministic": 4,
        "attested": 3,
        "inferred": 2,
        "hypothesized": 1,
    }

    update_cypher = """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        SET r.invalid_at = $invalid_at,
            r.expired_at = $now,
            r.superseded_by_uuid = $superseded_by
    """

    for (pred, subj), group in buckets.items():
        distinct_objects = {r["object"] for r in group}
        if len(distinct_objects) < 2:
            continue  # all claims agree on the object; no supersession

        # Find the latest by valid_at; tie-break by strength
        def sort_key(r: dict[str, Any]) -> tuple[Any, int]:
            return (r["valid_at"], strength_rank.get(r["strength"], 0))

        sorted_group = sorted(group, key=sort_key)
        newest = sorted_group[-1]
        # If the second-newest has the same valid_at AND different object,
        # it's an equal-time contradiction — leave live for conflict detector
        if len(sorted_group) >= 2:
            second_newest = sorted_group[-2]
            same_time = second_newest["valid_at"] == newest["valid_at"]
            different_object = second_newest["object"] != newest["object"]
            if same_time and different_object:
                continue  # let conflict detector handle

        # Invalidate everything older with a different object
        for r in group:
            if r["uuid"] == newest["uuid"]:
                continue
            if r["object"] == newest["object"]:
                continue  # corroborating, keep live
            if r["valid_at"] >= newest["valid_at"]:
                continue  # not older
            async with driver.session() as s:
                await s.run(
                    update_cypher,
                    uuid=r["uuid"],
                    invalid_at=newest["valid_at"],
                    now=now(),
                    superseded_by=newest["uuid"],
                )
            invalidated += 1
            audit.append(
                {
                    "superseded_uuid": r["uuid"],
                    "superseding_uuid": newest["uuid"],
                    "predicate": pred,
                    "subject": subj,
                    "reason": "newer_distinct_object",
                }
            )

    return {"invalidated": invalidated, "audit": audit}


# ===========================================================================
# Family conflict detection (adapted from family_conflict_detection.py)
# ===========================================================================


async def detect_family_conflicts(driver: AsyncDriver, pot: str) -> list[dict[str, Any]]:
    """For each (predicate, subject) bucket with multiple LIVE claims pointing
    at different objects, surface a conflict. Create a QualityIssue node
    linked to the contesting edges so agents can list them.

    Returns the list of conflicts created in this run.
    """
    cypher_live = """
        MATCH (a:Entity)-[r:RELATES_TO {group_id: $pot}]->(b:Entity)
        WHERE r.invalid_at IS NULL
        RETURN r.uuid AS uuid, r.name AS predicate,
               a.entity_key AS subject, b.entity_key AS object,
               r.valid_at AS valid_at, r.evidence_strength AS strength
    """
    async with driver.session() as s:
        rs = await s.run(cypher_live, pot=pot)
        rows = [dict(rec) async for rec in rs]

    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["predicate"], r["subject"])].append(r)

    conflicts_created = []
    for (pred, subj), group in buckets.items():
        distinct_objects = {r["object"] for r in group}
        if len(distinct_objects) < 2:
            continue

        # A conflict exists if multiple live claims point to different objects.
        # Pick a canonical conflict_id based on (pred, subj) — dedupe across runs.
        conflict_id = f"conflict:{pred}:{subj}"
        contesting = [r["uuid"] for r in group]

        upsert_cypher = """
            MERGE (q:Entity:QualityIssue {
                group_id: $pot, entity_key: $conflict_id
            })
            ON CREATE SET
                q.uuid = randomUUID(),
                q.code = 'predicate_family_conflict',
                q.severity = 'warning',
                q.status = 'open',
                q.kind = 'conflict',
                q.created_at = $now,
                q.predicate = $predicate,
                q.subject = $subject,
                q.message = $message
            SET q.contesting_edges = $contesting,
                q.distinct_objects = $objects,
                q.updated_at = $now
            RETURN q.entity_key AS key, q.uuid AS uuid
        """
        async with driver.session() as s:
            r = await s.run(
                upsert_cypher,
                pot=pot,
                conflict_id=conflict_id,
                now=now(),
                predicate=pred,
                subject=subj,
                message=(
                    f"{len(distinct_objects)} live claims about "
                    f"({subj}, {pred}, ?) with different objects: "
                    f"{sorted(distinct_objects)}"
                ),
                contesting=contesting,
                objects=sorted(distinct_objects),
            )
            record = await r.single()
            assert record is not None
            conflicts_created.append(
                {
                    "conflict_id": record["key"],
                    "predicate": pred,
                    "subject": subj,
                    "objects": sorted(distinct_objects),
                    "contesting_edge_count": len(contesting),
                }
            )

    return conflicts_created


async def list_open_conflicts(driver: AsyncDriver, pot: str) -> list[dict[str, Any]]:
    cypher = """
        MATCH (q:QualityIssue {group_id: $pot, kind: 'conflict', status: 'open'})
        RETURN q.entity_key AS conflict_id, q.predicate AS predicate,
               q.subject AS subject, q.distinct_objects AS objects,
               q.message AS message
    """
    async with driver.session() as s:
        rs = await s.run(cypher, pot=pot)
        return [dict(rec) async for rec in rs]


# ===========================================================================
# Embeddings (OpenAI direct — no Graphiti embedder)
# ===========================================================================


async def embed_text(text: str) -> list[float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    r = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding


async def ensure_indexes(driver: AsyncDriver) -> None:
    queries = [
        "CREATE INDEX entity_group_key IF NOT EXISTS "
        "FOR (n:Entity) ON (n.group_id, n.entity_key)",
        "CREATE INDEX claim_group_name IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)",
        "CREATE INDEX claim_group_invalid IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.invalid_at)",
        "CREATE INDEX episode_uuid IF NOT EXISTS "
        "FOR (e:Episodic) ON (e.uuid)",
        "CREATE INDEX episode_group IF NOT EXISTS "
        "FOR (e:Episodic) ON (e.group_id)",
        """
        CREATE VECTOR INDEX claim_fact_embeddings IF NOT EXISTS
        FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
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
            await s.run(q)


async def cleanup(driver: AsyncDriver, pot: str) -> None:
    async with driver.session() as s:
        await s.run("MATCH (n {group_id: $pot}) DETACH DELETE n", pot=pot)


# ===========================================================================
# Tests
# ===========================================================================


async def t10_episode_persistence(driver: AsyncDriver) -> dict[str, str]:
    step("T10 — Episode persistence without Graphiti")
    ref_time = now() - timedelta(hours=6)
    e1 = Episode(
        name="github.pr.merged.123",
        body="PR #123 merged: Add rate limiting to auth endpoint. Alice approved. Touched app/auth/middleware.py.",
        source_description="github:webhook:pull_request",
        reference_time=ref_time,
    )
    e2 = Episode(
        name="linear.issue.AUTH-42.closed",
        body="Issue AUTH-42 closed: Login flow fails on emails with + character. Root cause: URL decoding. Fixed in PR #123.",
        source_description="linear:webhook:issue",
        reference_time=ref_time + timedelta(hours=1),
    )
    e3 = Episode(
        name="incident.2026-05-19.auth-latency",
        body="Incident: auth-svc P95 latency spiked to 800ms at 03:00 UTC. Cause: db connection pool exhaustion. Mitigated by raising pool size; durable fix in PR #124.",
        source_description="pagerduty:incident",
        reference_time=ref_time + timedelta(hours=12),
    )
    u1 = await write_episode(driver, POT, e1)
    u2 = await write_episode(driver, POT, e2)
    u3 = await write_episode(driver, POT, e3)
    assert_true(all([u1, u2, u3]), "wrote 3 episodes with uuids")
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Episodic {group_id: $pot}) RETURN count(e) AS c", pot=POT
        )
        record = await r.single()
        assert record is not None
        c = record["c"]
    assert_eq(c, 3, "episode count")
    # Confirm embeddings populated
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Episodic {group_id: $pot}) "
            "WHERE e.body_embedding IS NULL RETURN count(e) AS c",
            pot=POT,
        )
        record = await r.single()
        assert record is not None
        c_null = record["c"]
    assert_eq(c_null, 0, "all episodes have embeddings (no Graphiti embedder needed)")
    return {"pr": u1, "issue": u2, "incident": u3}


async def t11_multi_label_entity(driver: AsyncDriver) -> None:
    step("T11 — Multi-label entity (Service + Activity)")
    # In the v2 ontology, rich change events carry both their canonical
    # label AND the Activity rollup label. Confirm this works.
    e = Entity(
        entity_key="deploy:auth-svc:v2.1.3:prod",
        labels=("Deployment", "Activity"),
        name="auth-svc v2.1.3 → prod",
        properties={"version": "v2.1.3", "environment": "prod"},
    )
    await write_entity(driver, POT, e)

    # Verify queryable as either label
    for label in ["Deployment", "Activity", "Entity"]:
        async with driver.session() as s:
            r = await s.run(
                f"MATCH (n:`{label}` {{group_id: $pot, entity_key: $key}}) "
                "RETURN count(n) AS c",
                pot=POT,
                key=e.entity_key,
            )
            record = await r.single()
            assert record is not None
            c = record["c"]
        assert_eq(c, 1, f"queryable as :{label}")


async def t12_environment_scoped(driver: AsyncDriver) -> None:
    step("T12 — Environment-scoped claim + env-filtered traversal")
    # The same Service+Component depends on different things in different envs.
    # Position B with environment as an edge property handles this cleanly.
    await write_entity(driver, POT, Entity("service:auth-svc", ("Service",), "auth-svc"))
    await write_entity(driver, POT, Entity("dep:stripe-live", ("Dependency",), "stripe-live"))
    await write_entity(driver, POT, Entity("dep:stripe-test", ("Dependency",), "stripe-test"))
    await write_entity(driver, POT, Entity("dep:mock-stripe", ("Dependency",), "mock-stripe"))

    base_args = dict(
        subject_key="service:auth-svc",
        predicate="USES",
        source_system="k8s-scanner",
        evidence_strength="deterministic",
        valid_at=now() - timedelta(hours=1),
    )
    await write_claim(driver, POT, Claim(
        **base_args, object_key="dep:stripe-live", environment="prod",
        source_ref="k8s/prod/auth-deployment.yaml@abc",
        fact="auth-svc uses stripe-live adapter in prod",
    ))
    await write_claim(driver, POT, Claim(
        **base_args, object_key="dep:stripe-test", environment="staging",
        source_ref="k8s/staging/auth-deployment.yaml@abc",
        fact="auth-svc uses stripe-test adapter in staging",
    ))
    await write_claim(driver, POT, Claim(
        **base_args, object_key="dep:mock-stripe", environment="local",
        source_ref="docker-compose.local.yaml@abc",
        fact="auth-svc uses mock-stripe adapter in local",
    ))

    # Query: in prod, what does auth-svc use?
    cypher = """
        MATCH (a:Entity {group_id: $pot, entity_key: $subject})
              -[r:RELATES_TO {name: 'USES'}]->(b:Entity)
        WHERE r.environment = $env AND r.invalid_at IS NULL
        RETURN b.entity_key AS dep
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=POT, subject="service:auth-svc", env="prod")
        prod_deps = [rec["dep"] async for rec in r]
    assert_eq(prod_deps, ["dep:stripe-live"], "prod USES = stripe-live")

    # Query: what's different between prod and staging?
    diff_cypher = """
        MATCH (a:Entity {group_id: $pot, entity_key: $subject})
              -[r1:RELATES_TO {name: 'USES', environment: $env_a}]->(b1)
        WHERE r1.invalid_at IS NULL
        WITH a, collect(b1.entity_key) AS in_a
        MATCH (a)-[r2:RELATES_TO {name: 'USES', environment: $env_b}]->(b2)
        WHERE r2.invalid_at IS NULL
        WITH in_a, collect(b2.entity_key) AS in_b
        RETURN [x IN in_a WHERE NOT x IN in_b] AS only_a,
               [x IN in_b WHERE NOT x IN in_a] AS only_b
    """
    async with driver.session() as s:
        r = await s.run(
            diff_cypher,
            pot=POT,
            subject="service:auth-svc",
            env_a="prod",
            env_b="staging",
        )
        record = await r.single()
    assert record is not None
    assert_eq(set(record["only_a"]), {"dep:stripe-live"},
              "prod-only USES = stripe-live")
    assert_eq(set(record["only_b"]), {"dep:stripe-test"},
              "staging-only USES = stripe-test")


async def t13_rescan_idempotency(driver: AsyncDriver) -> None:
    step("T13 — Re-scan idempotency")
    await write_entity(driver, POT, Entity("service:rescan-a", ("Service",), "rescan-a"))
    await write_entity(driver, POT, Entity("service:rescan-b", ("Service",), "rescan-b"))

    base = Claim(
        subject_key="service:rescan-a",
        predicate="DEPENDS_ON",
        object_key="service:rescan-b",
        source_system="k8s-scanner",
        source_ref="k8s/rescan/policy.yaml@v1",
        evidence_strength="deterministic",
        fact="rescan-a depends on rescan-b (first scan)",
        valid_at=now() - timedelta(hours=2),
    )
    uuid1 = await write_claim(driver, POT, base)

    # Re-scan: same source_ref but updated fact text and later valid_at
    rescan = Claim(
        subject_key="service:rescan-a",
        predicate="DEPENDS_ON",
        object_key="service:rescan-b",
        source_system="k8s-scanner",
        source_ref="k8s/rescan/policy.yaml@v1",
        evidence_strength="deterministic",
        fact="rescan-a depends on rescan-b (refreshed scan, updated comment)",
        valid_at=now() - timedelta(minutes=5),
    )
    uuid2 = await write_claim(driver, POT, rescan)

    assert_eq(uuid1, uuid2, "same edge uuid after re-scan with same source_ref")

    cypher = """
        MATCH ()-[r:RELATES_TO {group_id: $pot, source_ref: $sref}]->()
        RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=POT, sref="k8s/rescan/policy.yaml@v1")
        rows = [dict(rec) async for rec in r]
    assert_eq(len(rows), 1, "exactly one edge for re-scanned source_ref")
    assert_true(
        "refreshed scan" in rows[0]["fact"],
        "edge updated in place with new fact text",
        f"fact={rows[0]['fact']!r}",
    )


async def t14_episode_provenance(driver: AsyncDriver, episodes: dict[str, str]) -> None:
    step("T14 — Episode → claim provenance backlink")
    # Write a claim derived from the PR episode
    pr_ep = episodes["pr"]
    await write_entity(driver, POT, Entity("pr:auth/repo:123", ("PullRequest", "Activity"),
                                            "PR #123", {"pr_number": 123}))
    await write_entity(driver, POT, Entity("code:app/auth/middleware.py",
                                            ("CodeAsset",), "middleware.py"))
    await write_claim(driver, POT, Claim(
        subject_key="pr:auth/repo:123",
        predicate="MODIFIED",
        object_key="code:app/auth/middleware.py",
        source_system="github-webhook",
        source_ref=f"episode:{pr_ep}",
        evidence_strength="deterministic",
        fact="PR #123 modified app/auth/middleware.py",
        valid_at=now() - timedelta(hours=5),
        episode_uuid=pr_ep,
    ))

    # Navigate from claim → source episode body
    cypher = """
        MATCH ()-[r:RELATES_TO {group_id: $pot, name: 'MODIFIED'}]->()
        MATCH (e:Episodic {uuid: r.episode_uuid})
        RETURN e.body AS body, e.source_description AS source
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=POT)
        record = await r.single()
    assert record is not None
    assert_true(
        "PR #123" in record["body"],
        "claim → episode body navigation",
        f"source={record['source']}",
    )


async def t15_temporal_supersession(driver: AsyncDriver) -> None:
    step("T15 — Temporal auto-supersession (full mechanic)")

    await write_entity(driver, POT, Entity("svc:sup-a", ("Service",), "sup-a"))
    await write_entity(driver, POT, Entity("svc:sup-b", ("Service",), "sup-b"))
    await write_entity(driver, POT, Entity("svc:sup-c", ("Service",), "sup-c"))

    t_old = now() - timedelta(hours=24)
    t_new = now() - timedelta(hours=1)

    # Older: A DEPENDS_ON B
    await write_claim(driver, POT, Claim(
        subject_key="svc:sup-a", predicate="DEPENDS_ON",
        object_key="svc:sup-b",
        source_system="k8s-scanner",
        source_ref="k8s/sup/v1.yaml",
        evidence_strength="deterministic",
        fact="A depends on B (old)",
        valid_at=t_old,
    ))
    # Newer: A DEPENDS_ON C (different object)
    await write_claim(driver, POT, Claim(
        subject_key="svc:sup-a", predicate="DEPENDS_ON",
        object_key="svc:sup-c",
        source_system="k8s-scanner",
        source_ref="k8s/sup/v2.yaml",
        evidence_strength="deterministic",
        fact="A depends on C (new)",
        valid_at=t_new,
    ))

    result = await apply_predicate_family_supersede(driver, POT)
    assert_eq(result["invalidated"], 1, "older A→B claim invalidated")

    # Confirm A→B has invalid_at set; A→C is live
    cypher = """
        MATCH (a:Entity {entity_key: 'svc:sup-a'})
              -[r:RELATES_TO {name: 'DEPENDS_ON'}]->(b:Entity)
        WHERE r.group_id = $pot
        RETURN b.entity_key AS obj, r.invalid_at IS NULL AS live
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=POT)
        states = {rec["obj"]: rec["live"] async for rec in r}
    assert_eq(states["svc:sup-b"], False, "A→B no longer live")
    assert_eq(states["svc:sup-c"], True, "A→C is live")


async def t16_family_conflict(driver: AsyncDriver) -> None:
    step("T16 — Family-conflict QualityIssue creation (equal-time contradiction)")

    await write_entity(driver, POT, Entity("svc:conf-x", ("Service",), "conf-x"))
    await write_entity(driver, POT, Entity("svc:conf-y", ("Service",), "conf-y"))
    await write_entity(driver, POT, Entity("svc:conf-z", ("Service",), "conf-z"))

    # Two claims at the SAME valid_at with the same strength → equal-time
    # contradiction → conflict (not supersession).
    same_time = now() - timedelta(hours=3)
    await write_claim(driver, POT, Claim(
        subject_key="svc:conf-x", predicate="OWNED_BY",
        object_key="svc:conf-y",
        source_system="codeowners-scanner",
        source_ref="repos/conf/CODEOWNERS@abc",
        evidence_strength="attested",
        fact="conf-x owned by conf-y per CODEOWNERS",
        valid_at=same_time,
    ))
    await write_claim(driver, POT, Claim(
        subject_key="svc:conf-x", predicate="OWNED_BY",
        object_key="svc:conf-z",
        source_system="slack-discussion",
        source_ref="slack:eng-platform:1700",
        evidence_strength="attested",
        fact="conf-x owned by conf-z per Slack discussion",
        valid_at=same_time,
    ))

    # Supersede pass should NOT invalidate (equal time)
    supersede_result = await apply_predicate_family_supersede(driver, POT)
    # Supersede pass may invalidate other things from prior tests; check our pair
    cypher_live = """
        MATCH ()-[r:RELATES_TO {name: 'OWNED_BY', subject_key: 'svc:conf-x'}]->()
        WHERE r.group_id = $pot AND r.invalid_at IS NULL
        RETURN count(r) AS c
    """
    async with driver.session() as s:
        r = await s.run(cypher_live, pot=POT)
        record = await r.single()
    assert record is not None
    assert_eq(record["c"], 2, "both equal-time claims still live")

    # Now run the conflict detector
    conflicts = await detect_family_conflicts(driver, POT)
    our_conflict = next(
        (c for c in conflicts if c["predicate"] == "OWNED_BY" and c["subject"] == "svc:conf-x"),
        None,
    )
    assert_true(our_conflict is not None, "conflict surfaced for conf-x OWNED_BY")
    assert_true(our_conflict is not None, "")
    if our_conflict is not None:
        assert_eq(set(our_conflict["objects"]), {"svc:conf-y", "svc:conf-z"},
                  "conflict contesting objects")

    # Confirm agent can list it
    open_conflicts = await list_open_conflicts(driver, POT)
    our_open = next(
        (c for c in open_conflicts if c["predicate"] == "OWNED_BY"
         and c["subject"] == "svc:conf-x"), None,
    )
    assert_true(our_open is not None, "list_open_conflicts surfaces it")


async def t17_scale(driver: AsyncDriver) -> None:
    step("T17 — Scale test (1000 entities, 5000 claims, traversal latency)")
    # Use a separate scale pot so we don't pollute the test pot
    scale_pot = "pot:position-b-scale"
    await cleanup(driver, scale_pot)

    n_entities = 1000
    n_claims = 5000
    rng = random.Random(42)

    t0 = time.perf_counter()
    # Write entities in a single transaction
    async with driver.session() as s:
        await s.run(
            """
            UNWIND $entities AS e
            MERGE (n:Entity {group_id: $pot, entity_key: e.entity_key})
            SET n:Service, n.name = e.name, n.uuid = randomUUID(),
                n.created_at = $now
            """,
            pot=scale_pot,
            now=now(),
            entities=[
                {"entity_key": f"svc:s{i:04d}", "name": f"service-{i:04d}"}
                for i in range(n_entities)
            ],
        )
    t_entities = time.perf_counter() - t0

    # Generate claims
    claims_data = []
    for _ in range(n_claims):
        i = rng.randrange(n_entities)
        # Each service depends on 1-5 others; avoid self-loops
        j = (i + 1 + rng.randrange(n_entities - 1)) % n_entities
        valid_at = now() - timedelta(hours=rng.randrange(1, 720))
        claims_data.append(
            {
                "subject_key": f"svc:s{i:04d}",
                "object_key": f"svc:s{j:04d}",
                "source_ref": f"k8s/scale/{rng.randrange(10000):05d}",
                "fact": f"service-{i:04d} depends on service-{j:04d}",
                "valid_at": valid_at,
            }
        )

    t1 = time.perf_counter()
    # Bulk insert claims
    async with driver.session() as s:
        await s.run(
            """
            UNWIND $claims AS c
            MATCH (a:Entity {group_id: $pot, entity_key: c.subject_key})
            MATCH (b:Entity {group_id: $pot, entity_key: c.object_key})
            MERGE (a)-[r:RELATES_TO {
                group_id: $pot, name: 'DEPENDS_ON',
                subject_key: c.subject_key, object_key: c.object_key,
                source_ref: c.source_ref
            }]->(b)
            ON CREATE SET r.uuid = randomUUID(), r.created_at = $now,
                          r.invalid_at = null
            SET r.valid_at = c.valid_at,
                r.source_system = 'k8s-scanner',
                r.evidence_strength = 'deterministic',
                r.fact = c.fact,
                r.observed_at = $now
            """,
            pot=scale_pot,
            now=now(),
            claims=claims_data,
        )
    t_claims = time.perf_counter() - t1

    print(f"  loaded {n_entities} entities in {t_entities*1000:.0f}ms, "
          f"{n_claims} claims in {t_claims*1000:.0f}ms")

    # Confirm the actual edge count (MERGE may have collapsed duplicates)
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[r:RELATES_TO {group_id: $pot}]->() RETURN count(r) AS c",
            pot=scale_pot,
        )
        record = await r.single()
    assert record is not None
    actual_claims = record["c"]
    print(f"  actual edge count after dedup: {actual_claims}")

    # Latency: traversal at depth 1, 2, 3 from a random root
    root = "svc:s0042"
    for depth in [1, 2, 3]:
        cypher = f"""
            MATCH (root:Entity {{group_id: $pot, entity_key: $root}})
                  -[edges:RELATES_TO*1..{depth}]->(reached:Entity)
            WHERE all(r IN edges WHERE r.name = 'DEPENDS_ON' AND r.invalid_at IS NULL)
            RETURN count(DISTINCT reached) AS n
        """
        ts = time.perf_counter()
        async with driver.session() as s:
            r = await s.run(cypher, pot=scale_pot, root=root)
            record = await r.single()
        elapsed = (time.perf_counter() - ts) * 1000
        assert record is not None
        n_reached = record["n"]
        print(f"  depth={depth}: reached {n_reached:>4d} entities in {elapsed:>5.1f}ms")
        assert_true(elapsed < 500, f"depth-{depth} traversal under 500ms",
                    f"{elapsed:.1f}ms")

    # Cleanup scale pot
    await cleanup(driver, scale_pot)


async def t18_episode_semantic_search(
    driver: AsyncDriver, episodes: dict[str, str]
) -> None:
    step("T18 — Episode semantic search (Graphiti.search() replacement)")
    # Find the episode whose body matches "database connection issue"
    query = "database connection pool exhausted causing latency"
    qe = await embed_text(query)

    # Wait for vector index to be ready
    await asyncio.sleep(1.0)

    cypher = """
        CALL db.index.vector.queryNodes(
            'episode_body_embeddings', 3, $query_embedding
        ) YIELD node AS e, score
        WHERE e.group_id = $pot
        RETURN e.uuid AS uuid, e.name AS name, e.body AS body, score
        ORDER BY score DESC
    """
    async with driver.session() as s:
        r = await s.run(cypher, query_embedding=qe, pot=POT)
        hits = [dict(rec) async for rec in r]

    print(f"  query: {query!r}")
    for h in hits[:3]:
        print(f"    score={h['score']:.3f}  {h['name']}")
    assert_true(len(hits) > 0, "episode vector search returned hits")
    # Top hit should be the incident episode
    assert_eq(hits[0]["uuid"], episodes["incident"], "top hit is the incident episode")


async def t19_no_graphiti_import() -> None:
    step("T19 — Confirm graphiti_core is not imported in this module")
    # Walk this module's imports + transitively confirm none touched graphiti_core
    import sys

    graphiti_modules = [m for m in sys.modules.keys() if "graphiti" in m.lower()]
    assert_eq(graphiti_modules, [], "no graphiti modules loaded")
    # Belt-and-braces: check actual import statements (must match
    # start-of-line `import graphiti...` or `from graphiti... import`).
    import re

    with open(__file__) as f:
        source = f.read()
    import_re = re.compile(
        r"^(\s*)(import\s+graphiti|from\s+graphiti)", re.MULTILINE
    )
    matches = import_re.findall(source)
    assert_eq(matches, [], "no actual graphiti import statements in module")


# ===========================================================================
# Main
# ===========================================================================


async def main() -> None:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        print(f"{FAIL} Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{FAIL} Missing OPENAI_API_KEY (this POC embeds episodes + facts)")
        sys.exit(1)

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        header("Position B + Graphiti-free POC")
        print(f"  pot:    {POT}")
        print(f"  neo4j:  {uri}")
        print(f"  imports: only `neo4j` + `openai` + `dotenv`")

        header("Setup")
        await cleanup(driver, POT)
        await ensure_indexes(driver)
        ok("cleared prior data; indexes ensured (incl. native vector indexes)")

        header("Tests")
        await t19_no_graphiti_import()
        episodes = await t10_episode_persistence(driver)
        await t11_multi_label_entity(driver)
        await t12_environment_scoped(driver)
        await t13_rescan_idempotency(driver)
        await t14_episode_provenance(driver, episodes)
        await t15_temporal_supersession(driver)
        await t16_family_conflict(driver)
        await t18_episode_semantic_search(driver, episodes)
        await t17_scale(driver)

        header("Result")
        ok("All 10 deep-POC tests passed against bare Neo4j 5.x + OpenAI.\n  Graphiti was not imported.\n")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
