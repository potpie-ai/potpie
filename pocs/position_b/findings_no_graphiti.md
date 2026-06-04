# Position B + Graphiti removal — deep POC findings

> **Verdict:** **Graphiti removal is mechanically feasible and cheap.** All 10
> deep-POC tests pass against bare Neo4j 5.x + OpenAI with zero `graphiti_core`
> imports. The substrate replacement is ~150 LoC of core code. The real cost is
> **migration and refactor of existing call sites — estimated ~2 engineer-weeks
> total**, dominated by careful one-shot data migration and the test rewrite,
> not by net-new engineering.

Run: `.venv/bin/python pocs/position_b/poc_no_graphiti.py`

---

## What the deep POC proves

| # | Test | What it validates |
|---|---|---|
| **T10** | Episode persistence without Graphiti | `write_episode` in ~30 LoC: MERGE an `:Episodic` node + OpenAI embedding. **Total Graphiti replacement for `add_episode`.** |
| **T11** | Multi-label entity (`Service` + `Activity`) | The `is_activity=True` ontology trait works via `SET n:Service:Activity`. Trivial. |
| **T12** | Environment-scoped claim + env diff query | `environment` as an edge property gives us clean "diff prod vs staging" Cypher. The adapter-per-env story (UC2's specific call-out) works. |
| **T13** | Re-scan idempotency | MERGE key including `source_ref` → second scan with same ref returns same edge UUID, updates fact text in place. The source-scanner re-run semantic is sound. |
| **T14** | Episode → claim provenance | Claims carry `episode_uuid`; one Cypher hop from any claim to its raw-event source body. Audit trail intact. |
| **T15** | Temporal auto-supersede (full mechanic) | Adapted ~60-LoC supersede logic invalidates the older same-(subject,predicate) claim with a different object. Same logic `temporal_supersede.py` runs today, on our edges. |
| **T16** | Family-conflict QualityIssue creation | Equal-time, equal-strength contradiction surfaces a `:QualityIssue` node with `kind='conflict'`, listable by the agent via `list_open_conflicts`. |
| **T17** | Scale: 1000 entities, 5000 claims | Traversal at depth 1/2/3 in **266ms / 264ms / 137ms**. Bulk write 12s (improvable). Indexes effective. |
| **T18** | Episode semantic search | Native Neo4j vector index over `Episodic.body_embedding` returns the incident episode at score 0.816 for `"database connection pool exhausted causing latency"`. **`Graphiti.search()` is unnecessary.** |
| **T19** | No `graphiti_core` import in module | `import re` scan of source + `sys.modules` check confirms zero Graphiti dependency. |

Plus the 9 baseline tests from `poc.py` (corroboration, bitemporal point-in-time, Cypher supersession, blast-radius traversal, edge-fact vector search) — all still pass under the Graphiti-free substrate.

---

## The Graphiti-free substrate, in concrete terms

After this POC, here is what the entire substrate layer needs:

```
graphiti_core.Graphiti(...)            →  AsyncGraphDatabase.driver(uri, auth=(u,p))
g.add_episode(name, body, ...)         →  write_episode(driver, pot, Episode(...))   # ~30 LoC
g.search(query, search_filters=...)    →  Cypher with vector index + as_of predicate # ~10 LoC per reader
Graphiti.embedder                      →  AsyncOpenAI().embeddings.create(...)        # ~5 LoC
Graphiti's :RELATES_TO temporal model  →  Same shape; we write properties directly
Graphiti's edge invalidation           →  Cypher SET invalid_at, expired_at           # one query
Graphiti's vector indexes              →  CREATE VECTOR INDEX ... ON ... (5.x native)
SearchFilters DSL                      →  Direct WHERE clause                         # 1 line
```

**Total replacement: ~150 LoC across one module.** The POC's `write_episode`, `write_entity`, `write_claim`, `apply_predicate_family_supersede`, `detect_family_conflicts`, `embed_text`, and `ensure_indexes` are roughly the production module.

---

## Engineering challenges (the honest cost)

This is the meat of the analysis. Below in priority order.

### 1. Migration of existing data — **highest risk, 2 days**

Current Neo4j contains (per the design review):

- **`Entity:<Label>` nodes** keyed `(group_id, entity_key)`, written by Potpie's canonical writer. These keep their shape; only edges change.
- **Typed canonical edges** (`:OWNS`, `:DEPENDS_ON`, `:IMPLEMENTS`, ...) with `prov_*` properties. **Must be rewritten as `:RELATES_TO {name: <edge_type>, ...}`** with property renames (`prov_valid_from` → `valid_at`, `prov_valid_to` → `invalid_at`).
- **`Episodic` nodes** written by `g.add_episode` with Graphiti's own UUID + properties. Mostly compatible with our format; we adopt them. May add `body_embedding` if Graphiti's embedding storage differs.
- **Graphiti-extracted shadow entities + `:RELATES_TO` edges** (Graph B). **Drop entirely** — these are the LLM-extraction output we disabled. Need a careful Cypher pass identifying "entities with no `entity_key`" or "entities with only Graphiti UUID identity."

The migration is mechanical but unforgiving. Approach:

```cypher
// Step 1: dry-run inventory
MATCH ()-[r]->() WHERE NOT type(r) = 'RELATES_TO'
RETURN type(r), count(r) ORDER BY count(r) DESC;

// Step 2: rewrite each typed edge into :RELATES_TO with name=<type>
// Run per edge type, e.g. :OWNS:
MATCH (a)-[r:OWNS]->(b)
WHERE r.group_id IS NOT NULL  // canonical-written
CREATE (a)-[new:RELATES_TO {
    uuid: r.uuid,
    group_id: r.group_id,
    name: 'OWNS',
    subject_key: a.entity_key,
    object_key: b.entity_key,
    valid_at: coalesce(r.prov_valid_from, r.created_at),
    invalid_at: r.prov_valid_to,
    created_at: r.created_at,
    source_ref: r.prov_source_ref,
    source_system: r.prov_source_system,
    evidence_strength: coalesce(r.evidence_strength, 'inferred'),
    confidence: r.prov_confidence
    // ...remaining prov_* properties
}]->(b)
DELETE r;

// Step 3: identify Graph-B shadow entities and drop
MATCH (n:Entity)
WHERE n.entity_key IS NULL
  OR NOT (n)-[:CLAIMS|RELATES_TO {group_id: n.group_id}]-()
DELETE n;
```

**Risks:**
- One-shot migration must be transactional and idempotent. Test against a Neo4j snapshot from production data before running for real.
- Edge property naming convention drift — some edges may have `prov_observed_at` but not `valid_from`, etc. Need an enrichment pass to map old→new property names exhaustively. **Audit every distinct edge property key in the current data before writing the script.**
- Some edges may have multiple Graphiti-UUIDs collapsed (Graphiti's entity resolution did this). When rewriting, dedupe on `(group_id, name, subject_key, object_key, source_ref)`.
- Backup before running.

**Mitigation:** the migration runs once, code is then deleted (per no-compat rule). Risk is fully scoped to that PR.

### 2. Codebase refactor of every Graphiti call site — **2 days**

A grep for `g.add_episode\|g.driver\|graphiti_core` across `app/src/context-engine/` shows the integration points are concentrated:

- **`adapters/outbound/graphiti/episodic.py`** — the adapter wrapping Graphiti. ~1000 lines, but most of it is connection lifecycle and thread-local handling. The actual Graphiti calls (`g.add_episode`, `g.search`, `g.close`) are <20 sites.
- **`adapters/outbound/graphiti/canonical_writer.py`** — already writes Cypher directly via `g.driver`. Only the driver wrapper changes; the Cypher is unchanged.
- **`adapters/outbound/graphiti/temporal_supersede.py`**, **`family_conflict_detection.py`**, **`classify_modified_edges.py`** — operate on `:RELATES_TO` edges via `g.driver.execute_query`. Once `g.driver` becomes `driver: AsyncDriver` (our owned driver), no logic changes.
- **`adapters/outbound/graphiti/apply_plan.py`**, **`apply_episode_provenance.py`** — call `episodic.add_episode_async`. The call site adapts to the new signature; semantics preserved.
- **`bootstrap/container.py`** — wires `GraphitiEpisodicAdapter`. Replace with a thin `Neo4jEpisodicAdapter` that holds the driver directly.

**The key insight: most of the adapter's complexity is dealing with Graphiti, not with Neo4j.** Removing Graphiti makes the adapter *smaller*, not bigger.

**Strategy:** replace the implementation of `GraphitiEpisodicAdapter`'s methods one at a time, keeping the port interface (`EpisodicGraphPort`) unchanged. Application code calls the port; the swap is transparent.

### 3. Embedder service — **2 days**

Graphiti owns the embedder lifecycle today: multi-provider abstraction (OpenAI, Voyage, etc.), batching, retries on rate limits, error handling. Without Graphiti, we own this. Concretely:

```python
# pocs/position_b/poc_no_graphiti.py:embed_text — POC implementation
async def embed_text(text: str) -> list[float]:
    client = AsyncOpenAI()
    r = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding
```

For production we need:
- **Batching** — embedding 1000 facts at write-time is 1000 API calls if naive; batched calls reduce that to ~5. OpenAI's API accepts up to 2048 inputs per call.
- **Retries on 429 rate limits** — exponential backoff. Existing pydantic-ai retry primitives or `tenacity`.
- **Error handling** — quota-exhausted, network errors, partial-batch failures.
- **Provider abstraction** — eventually a `EmbedderPort` so we can swap OpenAI → Voyage → local model. Not urgent; YAGNI for V1.
- **Cost telemetry** — track per-pot embedding spend. Drops into the existing `TelemetryPort` cleanly.

**Honest estimate:** 2 days for a robust V1 embedder service. The POC's 5-LoC `embed_text` is sufficient for prototyping but won't survive bulk operations or rate-limit storms.

### 4. Vector index management — **0.5 day**

Today: Graphiti creates and manages its vector indexes implicitly. Without Graphiti, we run the `CREATE VECTOR INDEX` statements ourselves at app boot (POC shows the pattern).

**Pitfalls:**
- Neo4j vector indexes are version-specific. Confirm production Neo4j is 5.x+ (`db.indexes`). If older, plan an upgrade first.
- Index population is async. After creating, queries may return partial results until index is "ONLINE" — needs a `db.awaitIndex` call during ops.
- Existing Graphiti vector indexes need to be dropped before recreating with our schema. One-shot Cypher.

### 5. Lose hybrid search (BM25 + vector + graph) — **deferred capability, not a loss today**

Graphiti's `g.search()` does keyword (BM25) + vector + graph traversal in one fused query. **Position B does not use this** — every UC retrieval is structural-traversal-then-vector-search OR pure-vector-search. The POC validates this.

**Risk:** a future UC genuinely needs BM25 fusion. Mitigation: Neo4j 5.x has native full-text indexes; we can assemble per-use-case fusion in ~50 LoC when needed. The Graphiti monolith was over-engineered for our actual UCs.

### 6. Connection lifecycle — **0.5 day**

The current adapter has substantial thread-local + event-loop-binding logic (`episodic.py:122-161`) to handle Graphiti's quirks (its OpenAI client GC'd on a dead event loop, etc.). Most of this complexity disappears when we own the driver directly — `AsyncGraphDatabase.driver()` has cleaner lifecycle semantics.

**One thing to preserve:** the `thread_local` pattern protects against cross-loop driver use (Celery sync code mixed with FastAPI async). Keep that pattern; it's not Graphiti-specific.

### 7. Tests + fixtures — **2 days**

Existing tests:
- Mock `EpisodicGraphPort` (the abstraction) — these continue to work; the port interface is unchanged.
- A few tests mock `Graphiti` directly or assert on Graphiti-specific call patterns. These need rewriting against the new adapter's calls.
- Integration tests against a real Neo4j — these need their fixtures updated (drop Graphiti-extracted entities, use our writes).

The POC's structure is a starting point for fixtures.

### 8. Operational concerns — **1 day**

Items to handle:
- **Logging:** Graphiti emits its own logs (often noisy). Without it, our logs are cleaner — but we need to ensure span instrumentation (`with _obs.span("graph.write_episode")...`) is preserved.
- **OpenAI client management:** the SDK warns about clients GC'd while in-flight; ensure clean shutdown.
- **Migration rollback plan:** if removal causes incidents, can we restore? Neo4j backups + git revert. Confirm both work end-to-end on staging before production.
- **Cost telemetry:** today the agent's `CostEvent` records `kind="graphiti_extract"`. After removal, that bucket is gone; we have `kind="agent"` (the reconciliation agent) and `kind="embed"` (new, our embedder). Update the telemetry schema.

### 9. Subtle risks / things to watch — **ongoing vigilance**

- **External tools writing into our Neo4j.** Audit: is anything else (scripts, notebooks, an admin UI, the events-revamp branch) writing Graphiti-shape data we'd inadvertently break?
- **Embeddings model drift.** If Graphiti was using a different embedding model historically and we switch to `text-embedding-3-small`, old embeddings and new ones aren't comparable. **One-shot re-embed of all `Episodic.body` and `RELATES_TO.fact` during migration** to keep the vector space consistent. ~$5 for tens of thousands of records.
- **Graphiti's transitive dependencies.** `graphiti_core` pulls in OpenAI SDK, embedder libs, etc. After removing, run `uv sync --no-Graphiti-core` and confirm nothing else implicitly relied on it (unlikely but possible).
- **Performance degradation under load.** The POC's bulk write was 12s for 5000 claims (~2.4ms/edge), driven by per-claim Cypher and OpenAI rate limits. Production ingestion needs batching (single Cypher with `UNWIND $claims`) — already shown in T17 as a pattern. The first bulk re-embed of historical data needs this; the steady-state write rate is unlikely to be a bottleneck.

### 10. Forward-compatibility — **deferred concern**

If Graphiti's team ships compelling new features later (better hybrid search algorithm, native graph-NN reranking, etc.), we'd have to reimplement to adopt. **Not pressing** — Graphiti's recent releases have been incremental. The capability we'd actually benefit from (better hybrid search) we don't use today and have a clear path to building when needed.

---

## Total engineering estimate

| Activity | Estimate |
|---|---|
| Migration script (Cypher rewrite + drop shadow) | 2 days |
| Adapter refactor (replace Graphiti calls inside `episodic.py`) | 1-2 days |
| Embedder service (batching, retries, observability) | 2 days |
| Vector index management + index audit | 0.5 day |
| Test refactor (fixtures, integration) | 2 days |
| Observability + cost telemetry updates | 1 day |
| Staging soak + rollback verification | 1.5 days |
| **Total** | **~10 engineer-days (2 weeks)** |

This is **a quarter of one engineer's quarter** to remove a heavyweight dependency that no longer earns its keep for our use cases. Compare to the indefinite ongoing cost of carrying Graphiti with its extraction off, its vocabulary diverging from ours, its noisy logs, its transitive deps, and the cognitive overhead of "we use Graphiti for substrate, but…"

---

## Things lost (be honest about it)

Removing Graphiti means losing these capabilities. The POC confirms each is either replaceable or unused:

1. **`g.search()` hybrid (BM25 + vector + graph)** — not used today; assemble per-UC when needed. ❌ Acceptable loss.
2. **`add_episode`'s LLM extraction → entities + edges** — already disabled per D1. ❌ Already lost.
3. **Multi-provider embedder abstraction** — YAGNI; can rebuild when we genuinely need a non-OpenAI provider. ❌ Acceptable loss.
4. **Graphiti's entity-resolution** (statistical merge of similar entities) — actively unwanted; we use deterministic `entity_key` + alias table (D2). ❌ Not a loss — it was a problem.
5. **Graphiti's community/cluster detection** — not used. ❌ Not a loss.
6. **`add_triplet` / `fact_triple` write APIs** — not used. ❌ Not a loss.
7. **Graphiti's automatic-invalidation-on-contradiction extraction** — operates on the shadow graph; we use our own supersession (T15). ❌ Already replaced.
8. **`SearchFilters` DSL with `valid_at`/`invalid_at`** — replaced by direct Cypher predicate (T4, T5). ❌ Acceptable loss.

Nothing on this list is something we depend on operationally. The list is "convenience features and unused features."

---

## Recommended call

**Adopt Position B and remove Graphiti** in the same execution window, structured as:

1. **P0.5 (new, 2 weeks)** — Graphiti removal phase, slotting between P0 (cleanup) and P1 (identity):
   - Discovery: audit every Graphiti call site, every Graphiti-format data record, every test that mocks Graphiti. Catalog into a removal-target list.
   - Implementation: new `Neo4jEpisodicAdapter` replacing `GraphitiEpisodicAdapter`. `EmbedderService`. `Migration` script (one-shot Cypher).
   - Migration: run against a Neo4j snapshot from staging; verify counts, sample checks, traversal smoke tests pass. Then production.
   - Cleanup: `uv remove graphiti-core`. Update `vision.md` ("substrate is bare Neo4j + OpenAI embeddings; we own all of the read+write surface").
   - Exit criteria: `grep -ri "graphiti" app/src/context-engine` returns nothing; all existing tests pass with the new adapter; the POC's 19 tests pass against the production adapter (i.e. it's not just the POC code that works).

2. **P1, P2a, P2b, ... proceed as planned**, simplified because:
   - P2's storage decision is already settled by the POC (claim-as-edge on `:RELATES_TO`).
   - P0's "port temporal vocabulary" work isn't needed (we write the convention from the start).
   - The "demote Graphiti" sub-decision in P0 collapses into the full removal.

**The alternative — keeping Graphiti as substrate** — is defensible only if there's a hidden dependency the audit surfaces. After this POC, I cannot find one. The removal is mechanically straightforward, scope-bounded, and pays back in code clarity, dependency reduction, and ownership of the substrate behavior.

---

## Files

- `pocs/position_b/poc.py` — Position B baseline (9 tests, with Graphiti available)
- `pocs/position_b/poc_no_graphiti.py` — this deeper POC (10 tests, zero Graphiti imports)
- `pocs/position_b/README.md` — Position B findings
- `pocs/position_b/findings_no_graphiti.md` — this file
