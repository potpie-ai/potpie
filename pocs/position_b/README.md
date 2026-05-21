# Position B — POC findings (deep dive)

> **Verdict:** **Position B works end-to-end.** Every architectural claim
> verified against a real Neo4j 5.x + OpenAI stack in ~6 seconds of wall
> time, ~250 LoC of POC code, zero Graphiti API calls for the substrate
> behaviors we care about. The rebuild plan should adopt Position B as the
> default canonical edge shape.

Run: `.venv/bin/python pocs/position_b/poc.py` (after `pip install python-dotenv` if needed).

## What the POC proves

| # | Test | Architectural claim | Verdict |
|---|---|---|---|
| T1 | Write `:Entity` nodes keyed `(group_id, entity_key)` via direct Cypher | Deterministic identity (D2) is preserved unchanged | ✅ |
| T2 | Write `:RELATES_TO` edges with bitemporal + claim properties | The edge shape Position B adopts is writable directly via `g.driver` | ✅ |
| T3 | Two sources writing the same `(s, p, o)` land as two edges; belief derivation aggregates them | Corroboration works naturally via MERGE-key including `source_ref` | ✅ — 2 edges, `confidence=high` from `deterministic + attested` |
| T4 | Point-in-time via direct Cypher (`valid_at`/`invalid_at` predicate) | The bitemporal model is one Cypher predicate | ✅ — `as_of T-5d=0`, `T-36h=1`, `now=5` |
| T5 | Point-in-time via Graphiti's `SearchFilters` DSL | Graphiti's filter API is structurally compatible with our edges (same property names) | ✅ — `SearchFilters(valid_at=..., invalid_at=...)` constructs cleanly, same Cypher predicate |
| T6 | Newer claim stamps `invalid_at` on older same-`(s, p)` claim with different object | Supersession is one Cypher query, no Graphiti dependency | ✅ — 2 older claims invalidated, new belief flips, history preserved |
| T7 | `as_of T-12h` returns the old belief; `as_of now` returns the new one | True bitemporal correctness across supersession | ✅ — `users-svc` at T-12h, `billing-svc` at now |
| T8 | Variable-length `(:RELATES_TO*1..3 {name='DEPENDS_ON'})` traversal | UC2 blast-radius is tractable with property-filtered traversal | ✅ — **81ms at POC scale**, well under any latency concern |
| T9 | Embed `fact` text via OpenAI, index in Neo4j 5.x relationship vector index, similarity search | UC4 killer query (find by symptom signature) is one Cypher call | ✅ — `score=0.811` on the relevant fact, top hit correct |

## The shape — concretely

### Entities (unchanged from today)

```cypher
MERGE (n:Entity {group_id: $pot, entity_key: $entity_key})
SET n:`Service`, n.name = "auth-svc", n.uuid = randomUUID(), ...
```

`(group_id, entity_key)` MERGE keeps deterministic identity. D2's alias
layer overlays on top. **No change from today on the entity side.**

### Edges — the actual Position B move

```cypher
MERGE (a)-[r:RELATES_TO {
    group_id: $pot,
    name: $predicate,            -- "DEPENDS_ON", "OWNED_BY", "STORED_IN", ...
    subject_key: $subject_key,
    object_key: $object_key,
    source_ref: $source_ref       -- distinguishes corroborating claims
}]->(b)
ON CREATE SET
    r.uuid = randomUUID(),
    r.created_at = $now,
    r.expired_at = null,
    r.invalid_at = null
SET
    r.valid_at = $valid_at,        -- event time
    r.source_system = $source,
    r.evidence_strength = $strength,
    r.fact = $fact,                -- text for embedding + agent reading
    r.confidence = $confidence,
    r.observed_at = $now
```

**Every fact is one edge of this shape.** The MERGE key includes
`source_ref`, so two sources making the same claim produce two edges;
re-scanning the same source updates the existing edge in place
(idempotent). This is the **claim model from P2**: each `:RELATES_TO`
edge is one claim with full provenance.

### Supersession — five lines of Cypher

```cypher
MATCH (a {entity_key: $subject_key})-[r:RELATES_TO {name: $predicate}]->(b)
WHERE b.entity_key <> $new_object_key
  AND r.invalid_at IS NULL
  AND r.valid_at < $new_valid_at
SET r.invalid_at = $new_valid_at,
    r.expired_at = $now,
    r.superseded_by_uuid = $new_uuid
```

That's it. The same mechanism `temporal_supersede.py` runs today on
Graphiti's shadow edges — running here on edges we wrote ourselves.
**Position B inherits this for free.**

### Point-in-time — one predicate everywhere

```cypher
WHERE (r.valid_at IS NULL OR r.valid_at <= $as_of)
  AND (r.invalid_at IS NULL OR r.invalid_at > $as_of)
```

T4 ran this directly. T5 confirmed Graphiti's `SearchFilters(valid_at=...,
invalid_at=...)` DSL constructs the *same* predicate against the *same*
properties. Whether we call Graphiti's API or write the Cypher
ourselves, the data model and the queries are identical.

### Belief derivation — 30 lines of Python

```python
async def beliefs_for(driver, subject_key, predicate):
    # Aggregate live claims; rank candidate objects by max strength + corroboration
    ...
    score = max_strength + min(corroboration_count - 1, 3) * 0.5
    label = "high" if score >= 4 else "medium" if score >= 2.5 else ...
```

T3 confirmed the formula in practice: `deterministic(4) + attested(3,
corroboration bonus 0.5)` → score 4.5 → label `high`. After supersession
(T6), the same query returns `medium` because the new winning claim has
no corroboration yet.

This is the **D3 belief-derivation service in miniature**. The full
P2b implementation adds decay over time and a richer conflict surface,
but the substrate is here.

### UC4 — semantic search over claim facts

```cypher
CALL db.index.vector.queryRelationships(
    'claim_fact_embeddings', $top_k, $query_embedding
) YIELD relationship AS r, score
MATCH (a)-[r]->(b)
WHERE r.group_id = $pot AND r.invalid_at IS NULL
RETURN ..., score
```

T9 ran this with `text-embedding-3-small` embeddings over the `fact`
property. Query `"where is the database for auth-svc"` returned the
correct claim (`auth-svc stores data in postgres-auth via DATABASE_URL`)
at score 0.811. **Native Neo4j 5.x vector index on a relationship
property** — no Graphiti `g.search()` needed.

## What this means for the rebuild plan

Looking back at the rebuild plan I wrote, several phases shrink or
restructure significantly under Position B:

### P0 — Ground clearance

**Was:** rename `prov_valid_from`/`prov_valid_to` on canonical typed
edges. Keep typed edge labels. Demote Graphiti's extraction.

**Now:** migrate canonical typed edges to `:RELATES_TO {name: <type>}`
shape with the bitemporal property convention from the start.
Demote Graphiti's extraction — same.

**Why simpler:** there is no "port the temporal model" work because
the temporal model is already on `:RELATES_TO`. We just stop writing
typed labels and write `:RELATES_TO` with `name` instead.

### P2 — Claim model

**Was:** large phase deciding claim-as-edge vs claim-as-node, building
supersession, conflict, decay machinery from scratch.

**Now:**
- The "claim representation" question is **decided by the POC:
  claim-as-edge** on `:RELATES_TO`, with MERGE key including
  `source_ref` to support corroboration. Done.
- Supersession is **one Cypher query** (POC T6). The existing
  `temporal_supersede.py` runs on the same shape and can be adapted —
  or just kept after we point it at our edges instead of the shadow.
- Conflict detection runs the same way (`family_conflict_detection.py`
  already works on `:RELATES_TO`).
- Belief derivation is **30 lines** (POC code).
- Decay (the part the POC didn't implement) is one term in the score
  formula; pure addition.

P2 was the riskiest phase. The POC takes it out of the risk column.

### P5 — Event-driven activity layer

Unchanged. Activity claims are still `:RELATES_TO` edges with
`name="PERFORMED"` / `name="TOUCHED"` / etc.

### P7 — Ranking

The ranking formula (`strength × recency × scope-overlap ×
corroboration × semantic-similarity`) is computed per-candidate over
the claim edges. The POC already does a coarse version (`score =
max_strength + corroboration_bonus`). Adding semantic-similarity is the
T9 vector index score. **Everything the ranker needs is on the edge
properties.**

### P8 — Read surface

Each reader (UC1/UC2/UC3/UC4) is a thin Cypher wrapper over the claim
edges:

- **UC2** (`InfraTopologyReader`): blast-radius traversal (T8) +
  `as_of` filter + ranking. Maybe 100-150 LoC.
- **UC4** (`PriorBugsReader`): vector-index similarity (T9) + scope
  filter + verification ranking. Maybe 80-120 LoC.
- **UC1** (`CodingPreferencesReader`): claim edges with
  `name="POLICY_APPLIES_TO"`, filter by `code_scope` overlap, rank.
  Maybe 100 LoC.
- **UC3** (`TimelineReader`): `(:Activity)`-anchored window query over
  `name="PERFORMED"` / `"TOUCHED"` edges. Existing reader adapts
  easily.

## Where Graphiti now sits

Re-answering "what does Graphiti contribute" after the POC:

| Graphiti feature | Position B usage | Could replace? |
|---|---|---|
| Neo4j driver | Yes — `g.driver` for direct Cypher | Trivially (`neo4j-driver` directly) |
| `add_episode` for audit log | Yes — episodes are the raw-event provenance | Yes (small custom layer); not pressing |
| LLM extraction | **No** — disabled per D1; we don't read its output | N/A |
| `SearchFilters` DSL with bitemporal predicates | **Optional** — T5 proves it works on our edges, but T4 shows direct Cypher is equally clear | Yes; the DSL is small |
| Embeddings + vector search via `g.search()` | **No** — Neo4j 5.x native vector index works directly (T9) | N/A — we're already not using it |
| Hybrid search (BM25 + vector + graph) | Not used in POC | Yes if we ever need it; nothing built on it now |
| Temporal supersede / conflict machinery | **Adopted shape, not API** — our supersession runs the same Cypher pattern on our edges | N/A — we own the queries now |

**The honest read after this POC:** keeping Graphiti as a substrate
buys us:
1. The Neo4j driver wrapper (commodity).
2. `add_episode` for audit-log persistence (one-pager; trivially
   replaceable).
3. The `SearchFilters` DSL (which we don't even need — direct Cypher
   is as readable).
4. An LLM extraction pipeline we've turned off.

The case for replacing Graphiti gets *stronger* after this POC, not
weaker. Position B doesn't depend on any Graphiti feature; it adopts
Graphiti's *bitemporal property convention*, which is just four
property names. We could go further and **delete the Graphiti
dependency entirely** in a follow-up — but that's not necessary to
make Position B work, so the plan can defer it.

## Concerns / non-trivial things the POC surfaced

1. **Neo4j deprecation warning on relationship property index syntax.**
   `CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)` produces
   a deprecation hint in Neo4j 5.x — the modern syntax is relationship
   type indexes. Easy fix; noted for production.
2. **MERGE-key choice.** Including `source_ref` in the MERGE key means
   the same source re-scanning produces the same edge UUID (idempotent
   re-scan updates). That's correct for source-scanners (P4) but worth
   confirming for agent recordings (P6): when an agent records "I
   tried fix F and it didn't work," is each attempt a new edge
   (`source_ref=agent-run-uuid`) or an update? Recommend new edge per
   attempt; trivial.
3. **Traversal latency at scale.** 81ms at 5-edge POC. Neo4j with the
   composite index on `:RELATES_TO(group_id, name)` should scale well
   to thousands of edges; not validated at 10k+. Worth a load test in
   P4 once real scanner data flows.
4. **Edge embeddings cost.** OpenAI `text-embedding-3-small` at
   $0.02/1M tokens is cheap. ~$0.0001 per edge (5-50 token facts).
   1000 claims per pot ≈ $0.10. Re-embedding on `fact` change is
   trivial cost-wise; just need to invalidate the embedding when
   `fact` is updated.
5. **The `OWNED_BY` predicate** in the POC: today's ontology has it as
   the typed edge `:OWNS` going the other direction (`Person -[:OWNS]->
   Service`). Position B doesn't care about direction; what matters is
   the `(subject_key, predicate, object_key)` triple. Migration of
   existing typed edges to `:RELATES_TO` needs to settle these per
   ontology entry, but it's mechanical.

## What I'd commit to next

If the position is adopted:

1. **Update the rebuild plan** to make Position B the default. P0's
   migration becomes "typed edges → :RELATES_TO with names." P2's
   storage decision is settled. P7 and P9 phase scopes shrink.
2. **Add a Position B test scenario set to `bench-plan.md`**:
   corroboration → high; contradiction → conflict or supersession;
   `as_of` past returns history; UC2 blast-radius; UC4 symptom
   similarity. The POC's 9 tests are the seed.
3. **Write the migration plan for existing data.** Cypher to walk
   every existing typed-edge instance and rewrite as `:RELATES_TO
   {name: <edge type>}`. One-shot; no compat. Probably 50-100 LoC.
4. **Decide whether to keep Graphiti.** Not blocking — we can keep it
   as substrate for audit-log persistence and migrate off later if
   cost telemetry shows it's worth it. The decision is now data-driven,
   not faith-based.

## Files

- `pocs/position_b/poc.py` — the runnable POC (~700 LoC including tests
  + comments)
- `pocs/position_b/README.md` — this file
