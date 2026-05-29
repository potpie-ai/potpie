# Lightweight Local Graph DB (Neo4j alternative) — History

Research log for finding a lightweight, easy-to-setup graph database to run
**alongside** Neo4j for local / self-host development. Neo4j (JVM, multi-GB
heap) is heavy for laptops and quick-start onboarding. The goal is **not** to
rip Neo4j out — production stays on Neo4j — but to offer a small, in-memory
substitute that a user can `docker run` (or `pip install`) and be productive
in minutes.

Newest entries at the bottom.

Related logs: [`history.md`](history.md),
[`history-context-graph-observability.md`](history-context-graph-observability.md).

---

## 2026-05-28 — Kickoff + framing

**User ask:** Neo4j is resource-intensive for local running. Provide a
lightweight, in-memory graph DB users can set up easily (examples named:
FalkorDB, Memgraph). Keep Neo4j as the production substrate; add a local
alternative behind the existing seams. Research properly, document here.

**Branch:** `feat/local-context-engine`.

The hard part is not picking "the fastest small graph DB" — it's picking one
that the **existing code can actually talk to** without a rewrite that
outweighs the resource savings. So the first job was auditing exactly how
Neo4j is used today.

---

## 2026-05-28 — Audit: how Neo4j is wired today

There are **two independent graph workloads**, each with its own driver, its
own Cypher, and its own compatibility constraints. Any local-DB story has to
account for both.

### Workload A — Monolith "code knowledge graph" (parsing)

File: [`app/modules/parsing/graph_construction/code_graph_service.py`](app/modules/parsing/graph_construction/code_graph_service.py)

- Builds the graph in **NetworkX** (`nx.MultiDiGraph`) first, then bulk-writes
  to Neo4j via the **sync Bolt driver** (`neo4j.GraphDatabase`).
- **Uses APOC**: nodes are created with
  `CALL apoc.create.node(node.labels, node) YIELD node` — APOC is needed
  because the node labels are dynamic (`NODE` + one of
  `FILE/CLASS/FUNCTION/INTERFACE`). This is the single biggest portability
  blocker — APOC is a Neo4j-only plugin (compose loads it via
  `NEO4JLABS_PLUGINS: '["apoc"]'`).
- **Dynamic relationship types** via Python f-string interpolation
  (`CREATE (source)-[r:{rel_type} {...}]->(target)`).
- Composite index DDL: `CREATE INDEX ... IF NOT EXISTS FOR (n:NODE) ON
  (n.node_id, n.repoId)`.
- Vectors live in **Qdrant**, not Neo4j (separate hybrid dense+BM25+ColBERT
  collection per project). So the code-graph workload does *not* lean on
  Neo4j vector indexes.
- Other monolith consumers run hand-written Cypher against this graph:
  `kg_based_tools/*`, `code_query_tools/*` (neighbours, code-from-node-id,
  tags, probable-name, change detection).

### Workload B — Context-engine "context graph" (claims)

Dir: [`app/src/context-engine/adapters/outbound/graph/`](app/src/context-engine/adapters/outbound/graph/)

- Talks to Neo4j directly via the **Bolt driver, both sync
  (`GraphDatabase`) and async (`AsyncGraphDatabase`)**.
- Graph model (Position B): `(:Entity {group_id, entity_key})
  -[:RELATES_TO {name, subject_key, object_key, source_ref, valid_at,
  invalid_at, ...}]-> (:Entity)`. Bitemporal (event time + system time).
- Cypher features in use (see
  [`cypher.py`](app/src/context-engine/adapters/outbound/graph/cypher.py),
  [`neo4j_reader.py`](app/src/context-engine/adapters/outbound/graph/neo4j_reader.py),
  [`neo4j_writer.py`](app/src/context-engine/adapters/outbound/graph/neo4j_writer.py)):
  - `MERGE ... ON CREATE SET ... SET r += $props` (idempotent upserts).
  - Built-ins `randomUUID()`, `timestamp()`.
  - **Relationship property indexes**:
    `CREATE INDEX ... FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)`.
  - **Relationship VECTOR index** (Neo4j 5.x):
    `CREATE VECTOR INDEX claim_fact_embeddings ... FOR ()-[r:RELATES_TO]-()
    ON (r.fact_embedding)` with `vector.dimensions: 1536`,
    `vector.similarity_function: 'cosine'`. Already wrapped in try/except —
    code degrades to Python token-overlap scoring when the index is absent.
  - **`CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF $batch ROWS`** (Neo4j
    5.x batched-delete subquery) in `reset_pot`.
  - Map projection `r{.*}`, `labels(a)`, dynamic `SET e:{lbl}`.
- `fact_query` semantic search is currently the **Python fallback already**
  (token-overlap), not the native vector index — so the vector index is
  built but not yet on the hot read path. That lowers the bar for an
  alternative's vector support short-term.

### Connection seams (where a backend swap would plug in)

- **context-engine:** settings port exposes `neo4j_uri/user/password`
  ([`settings_env.py`](app/src/context-engine/adapters/outbound/settings_env.py),
  reads `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`). The container
  **hard-wires** the concrete classes —
  [`container.py:245`](app/src/context-engine/bootstrap/container.py#L245)
  `Neo4jGraphWriter(s)` and `:249` `Neo4jClaimQueryStore(s)`. There is a
  clean port pair to implement against: `GraphWriterPort` +
  `ClaimQueryPort`. No backend-selection branch exists yet — adding one is
  the natural insertion point.
- **monolith:** `config_provider.get_neo4j_config()` →
  `CodeGraphService(uri, user, password, db)` builds the Bolt driver in its
  constructor. Swapping means a sibling service or an injected driver.
- **graphiti-core** is a declared dependency (`requirements.txt`,
  `neo4j==6.1.0` pulled in transitively by graphiti). The graphiti adapter
  dir under context-engine is currently empty on this branch (only
  `__pycache__`), so graphiti is not on the active write path here — but its
  multi-backend driver is a relevant lever (see below).

### Local infra today

[`compose.yaml`](compose.yaml): `neo4j:latest` + APOC, ports 7474/7687,
`NEO4J_dbms_memory_transaction_total_max: 0` (unbounded txn memory). Driver
pins: context-engine `neo4j>=5.28.0`, monolith `neo4j==6.1.0`.

**Takeaway:** Workload B (context graph) is *mostly* portable Cypher over
Bolt. Workload A (code graph) is **APOC-bound** and is the real migration
cost. A "drop-in for local" that satisfies B but not A only solves half.

---

## 2026-05-28 — Candidate research (current as of 2026)

### FalkorDB (Redis module; GraphBLAS sparse matrices)

- **Footprint / setup:** the strongest fit for "small + easy." ~7x less RAM
  than Neo4j on the same dataset (sparse adjacency matrices, no JVM heap
  pre-allocation). Single container: `docker run -p 6379:6379
  falkordb/falkordb`. Also ships **FalkorDBLite** — an *embedded, zero-config*
  Python package (`pip install falkordblite`) that starts a local graph DB
  with no server to run. This directly matches "users set it up easily."
- **Query language:** openCypher — identical to Neo4j for basic
  MATCH/MERGE/CREATE/SET patterns, but a **subset**: no APOC, no
  `CALL { } IN TRANSACTIONS`, different DDL for indexes, its own vector index
  syntax.
- **Vectors:** built-in vector index + similarity (cosine / euclidean).
  Historically node-centric; relationship-vector parity vs Neo4j 5.x needs
  validation (but Workload B doesn't depend on it yet — Python fallback).
- **Protocol:** natively Redis (RESP) via the `falkordb-py` client. **Now
  also exposes Bolt** (opt-in `BOLT_PORT 7687`), so the existing `neo4j`
  driver *may* connect — but FalkorDB Bolt is newer than Memgraph's and its
  coverage of driver features needs a smoke test before relying on it.
- **graphiti-core support:** **yes, first-class** (graphiti's `FalkorDriver`,
  supported FalkorDB 1.1.2). So any graphiti-managed path is a config swap.

### Memgraph (C++ in-memory; Bolt-native)

- **Compatibility:** the strongest fit for "talk to existing code." Bolt is
  native, so the current `neo4j` Python driver connects as a drop-in — only
  the URI changes. Cypher is highly Neo4j-compatible for common patterns.
- **Footprint / setup:** much lighter than Neo4j (no JVM), in-memory by
  default with optional on-disk. Single container. Heavier RAM than FalkorDB
  per third-party benchmarks, but far below Neo4j.
- **Gaps that bite this codebase:**
  - **APOC**: `apoc.create.node(...)` (Workload A) is not available; Memgraph's
    equivalent is **MAGE** with different procedure names → monolith node
    creation must be rewritten regardless.
  - `CALL { } IN TRANSACTIONS` (Workload B reset) — not supported as-is.
  - Relationship **vector** index — Memgraph vector search is node-oriented;
    relationship-vector parity needs checking (again, not on the hot path).
  - DDL differences for relationship/composite indexes.
- **graphiti-core support:** **no** Memgraph driver in graphiti's supported
  set (Neo4j / FalkorDB / Kuzu / Neptune) → graphiti-managed paths can't use
  it.

### Kuzu (embedded, in-process; Cypher) — ⚠️ project risk

- Technically the *ideal* "local-first" shape: embedded like SQLite, no
  server, Cypher, built-in vector + full-text search, **graphiti-supported**
  (Kuzu 0.11.2).
- **But the project was archived in Oct 2025** — Kùzu Inc. wound down (talent
  acquired by Apple), GitHub repo archived. It now lives only through
  community forks (**Bighorn** by Kineviz, **Ladybug**, **RyuGraph**), none
  yet proven for long-term maintenance. **Not advisable as a fresh
  dependency** despite the great fit. Revisit if one fork clearly wins.

### Baseline — "just tune Neo4j"

- Neo4j Community with a small heap (`server.memory.heap.max_size=512m`,
  pagecache trimmed) + drop the unbounded-txn setting cuts local footprint
  with **zero code change** and 100% feature parity (APOC, vector, `CALL IN
  TRANSACTIONS` all keep working). Still JVM (hundreds of MB) and not
  "in-memory," but it's the no-migration floor to compare savings against.

### Quick matrix

| Criterion | FalkorDB | Memgraph | Kuzu (fork) | Neo4j (tuned) |
| --- | --- | --- | --- | --- |
| RAM footprint | ★★★ lowest | ★★ low | ★★★ embedded | ★ JVM |
| Easy setup | ★★★ (Lite = embed) | ★★ 1 container | ★★★ pip, embed | ★ 1 container |
| Existing Bolt driver works | partial (new Bolt) | ★★★ drop-in | ✗ own API | ★★★ native |
| openCypher/Cypher parity | subset | high | Cypher | full |
| APOC (Workload A) | ✗ rewrite | ✗ (MAGE) rewrite | ✗ rewrite | ★★★ native |
| `CALL IN TRANSACTIONS` | ✗ | ✗ | n/a | ★★★ |
| Vector (rel-level) | needs check | needs check | yes (node) | ★★★ native |
| graphiti-core backend | ★★★ yes | ✗ no | yes (archived) | ★★★ yes |
| Project health | ★★★ | ★★★ | ⚠️ archived | ★★★ |

---

## 2026-05-28 — Recommendation (draft, awaiting confirmation)

**Primary: FalkorDB as the lightweight local backend.** It uniquely satisfies
every stated requirement — smallest footprint, *embeddable* (FalkorDBLite =
literally `pip install`, no server), and it's a first-class graphiti backend.
The named-by-user shortlist (FalkorDB / Memgraph) plus the graphiti dependency
already in the tree all point the same way.

**Why FalkorDB over Memgraph**, despite Memgraph's better raw-Cypher
compatibility:
- Both force a rewrite of the **APOC** node-creation in Workload A — Memgraph's
  Bolt drop-in advantage doesn't save us there, because the blocker is
  procedures, not protocol.
- FalkorDB is graphiti-compatible; Memgraph is not. If/when the context graph
  moves onto graphiti's driver, FalkorDB is a config flag and Memgraph is a
  dead end.
- FalkorDB is dramatically lighter and can run *embedded*, which is the truest
  answer to "easy local setup."

**Why not Kuzu:** best technical fit, but archived/community-fork-only as of
Oct 2025 — too risky to adopt now.

**Honest caveats to validate before committing:**
1. FalkorDB Bolt support is newer — confirm whether the existing `neo4j`
   driver works against it, or whether the context-engine adapter must use
   `falkordb-py`. (A `FalkorDBClaimQueryStore` / `FalkorDBGraphWriter` behind
   the existing ports is the clean fallback either way.)
2. Rewrite Workload A's `apoc.create.node` to native multi-statement creation
   (one `CREATE`/`MERGE` per label combo, or set labels post-create). Needed
   for *any* non-Neo4j backend.
3. Re-express `CALL { } IN TRANSACTIONS` reset as a client-side batched
   `MATCH ... DETACH DELETE LIMIT $n` loop (portable across all three).
4. Vector parity is **not urgent** — Workload B already runs the Python
   token-overlap fallback and Workload A uses Qdrant. Native graph-vector can
   come later.

### Proposed shape (no code yet)

- Add a `GRAPH_DB_BACKEND` env (`neo4j` default | `falkordb`) to the
  context-engine settings port and a selection branch in
  [`container.py`](app/src/context-engine/bootstrap/container.py) — implement
  `FalkorDB*` adapters against the existing `GraphWriterPort` /
  `ClaimQueryPort` so nothing above the port changes.
- For the monolith, an analogous backend switch in `config_provider` +
  `CodeGraphService` (the APOC rewrite lands here).
- Add a `falkordb` service (or document FalkorDBLite) as an opt-in compose
  profile so the default Neo4j path is untouched; production stays on Neo4j.

**Status:** Research complete; recommendation drafted. Awaiting user
direction on (a) FalkorDB vs Memgraph and (b) scope — context-engine only
vs both workloads — before any implementation.

---

## 2026-05-28 — Scope narrowed: context-graph module only

User directed: evaluate **only the context-engine context-graph module**
(`app/src/context-engine/`), not the monolith code graph. This materially
changes the calculus, because the hardest blocker (APOC) lives entirely in
the monolith.

Exhaustive re-scan of `app/src/context-engine/` (excluding `.venv` + tests):

- **APOC: zero usages.** The APOC blocker is 100% the monolith's concern.
- **Neo4j-5-only syntax: one site** — `CALL {} IN TRANSACTIONS` in
  [`neo4j_writer.py:173`](app/src/context-engine/adapters/outbound/graph/neo4j_writer.py#L173)
  (`reset_pot`). Easily rewritten as a client-side batched
  `MATCH ... WITH n LIMIT $n DETACH DELETE n` loop.
- **Vector index: already optional** — the `CREATE VECTOR INDEX` in
  `cypher.py` is wrapped in try/except, and reads use a Python token-overlap
  fallback. So vector parity is not on the critical path.
- **graphiti: not on the active write path here** (adapter dir empty on this
  branch).
- **Raw Cypher is confined to 3 files** — `cypher.py`, `neo4j_writer.py`
  (async), `neo4j_reader.py` (sync) — all behind the `GraphWriterPort` /
  `ClaimQueryPort` ports. Backend is hard-wired at
  [`container.py:245`](app/src/context-engine/bootstrap/container.py#L245).

**Migration surface for this module = small:** the 3 graph files + a
backend-selection branch in `container.py` + a settings flag. Whatever
backend we pick, the same 3 Neo4j-isms must be patched (none of the
alternatives have them):

1. `randomUUID()` → generate client-side (`uuid4`).
2. `CALL {} IN TRANSACTIONS` reset → client-side batched delete loop.
3. Index DDL (node composite / relationship / vector) → backend-specific;
   vector index degrades gracefully if unsupported.

Because there is **no APOC here**, Memgraph's "Bolt drop-in" advantage is
actually realizable for this module (it was *not* for the whole repo) — which
reopens the field.

## 2026-05-28 — Expanded candidate survey (beyond FalkorDB/Memgraph/Kuzu)

Since the module talks to the DB only through ports, the query language is not
locked to Cypher — widening the field. Current (2026) state of additional
credible options:

- **ArcadeDB** — *new strong contender.* Open-source, multi-model, **tiny
  footprint**, speaks **Bolt (v3/4/4.4) + native OpenCypher passing 97.8% of
  the Cypher TCK** (Cypher 25). Rare combo: existing `neo4j` driver can
  connect *and* Cypher parity is far higher than FalkorDB's subset — so even
  `CALL {} IN TRANSACTIONS` / index DDL may work unchanged. Java (heavier than
  FalkorDB's Redis/C core, much lighter than Neo4j). Needs validation on
  relationship-vector + the batched-delete subquery against our exact queries.
- **Apache AGE** — PostgreSQL extension adding openCypher; actively maintained
  (release Feb 2026). Killer angle: **we already run Postgres in
  [`compose.yaml`](compose.yaml)**, so it's *zero new services* for local —
  the lightest setup story. Downsides: openCypher subset, queries wrapped in a
  SQL `cypher('graph', $$...$$)` call (adapter differs from a Bolt driver),
  extension build can be fiddly.
- **SurrealDB** — single Rust binary, runs **embedded / in-memory**, with
  graph + vector + temporal in one language (SurrealQL). Positions itself as
  "agent memory / context layer" — conceptually aligned with our bitemporal
  claims. But SurrealQL ≠ Cypher → adapter from scratch.
- **CozoDB** — embedded relational-graph-vector DB with **built-in
  time-travel** (natural fit for `valid_at`/`invalid_at`) + built-in HNSW
  vector. Datalog query language → non-Cypher rewrite. Smaller community.
- **NetworkX** — already a dependency (monolith builds graphs in it).
  Pure-Python, in-memory, lightest possible, zero setup — but no persistence,
  concurrency, or vector. Only viable as a throwaway single-process dev double.
- **Kuzu forks** (Bighorn / Ladybug / RyuGraph) — keep the embedded-Cypher fit
  alive after Kuzu's Oct-2025 archival, but none has proven longevity. Wait.

## 2026-05-28 — Refined recommendation (module-scoped)

Down to **three real contenders** for the context-graph module, each winning a
different axis:

| Priority | Best pick | Why |
| --- | --- | --- |
| Lightest + easiest end-user setup | **FalkorDB** | ~7x less RAM; embeddable via FalkorDBLite (`pip`, no server); graphiti-ready. Cost: new adapter on `falkordb-py`, openCypher subset |
| Minimal code change + full Cypher | **ArcadeDB** | Bolt drop-in (reuse `neo4j` driver) *and* ~full Cypher parity; small/embeddable. Cost: Java; validate vector + `CALL IN TRANSACTIONS` |
| Pure Bolt drop-in, mature | **Memgraph** | Bolt-native, in-memory; existing driver works. Cost: heavier than the other two; not a graphiti backend; some DDL/`randomUUID` patches |

**Lean:** if the stated goal stays *"lightweight + in-memory + easy local
setup,"* **FalkorDB** remains the pick (it wins that axis decisively, and the
module's clean ports keep the new-adapter cost bounded). **ArcadeDB** is the
one to actively evaluate as the alternative — it may beat Memgraph at the
"drop-in + light" game thanks to near-full Cypher over Bolt. **Apache AGE** is
the wildcard if "no new container" outranks everything (reuses existing
Postgres).

**Implementation shape (unchanged, no code yet):** `GRAPH_DB_BACKEND` setting
(`neo4j` default | chosen backend) + a branch at
[`container.py:245`](app/src/context-engine/bootstrap/container.py#L245);
new adapter(s) implementing the existing `GraphWriterPort` / `ClaimQueryPort`;
an opt-in local profile (compose service or embedded lib) so the default Neo4j
path and production stay untouched.

**Status:** Research complete and scoped to the context-graph module.
Decision pending between FalkorDB (lightest) and ArcadeDB (most compatible);
Memgraph as fallback. No code written.

---

## 2026-05-28 — Independent re-check + operational-overhead decision

Fresh pass against current docs and the actual context-engine shape:

- **ArcadeDB** is probably the lowest-code-change POC because it can run over
  Bolt with the official Neo4j drivers and advertises very high OpenCypher
  compatibility. For this codebase, that means the existing
  `Neo4jGraphWriter` / `Neo4jClaimQueryStore` path may need only connection
  settings plus small query/DDL patches. The catch: for our Python service it
  is still operationally a **separate ArcadeDB server/container**. Its
  embedded mode is primarily a Java in-process story, not a Python
  `pip install` local-dev story.
- **FalkorDB** is the best match for the original user problem: reduce local
  setup and laptop overhead. It can run as a small Redis-module container, and
  **FalkorDBLite** gives the cleanest embedded Python option: install a Python
  package and let the app control the local graph runtime. That does mean a
  purpose-built `FalkorDBGraphWriter` / `FalkorDBClaimQueryStore` is more
  likely than reusing the Neo4j driver path.
- **Memgraph** remains credible and mature, especially for Bolt-native
  compatibility, but it is also a separate server/container and does not beat
  ArcadeDB on compatibility or FalkorDB on setup simplicity.
- **Apache AGE** is appealing because it can reuse Postgres, but it changes
  the adapter model: Cypher is called through SQL (`cypher('graph', $$...$$)`),
  so it is not a drop-in for the current Neo4j-driver path.
- **Kuzu** stays out because the upstream project was archived in Oct 2025.

**Decision:** for the context-graph local backend, prefer **FalkorDB** because
the feature's stated value is operational simplicity for local/self-hosted
development, not minimum implementation diff. Keep **ArcadeDB** as the
compatibility fallback if FalkorDB's adapter/query work becomes too large.

### Slack comparison table

Shareable summary:

| Option | Setup overhead | Code-change risk | Current-driver reuse | Local-dev fit | Main concern | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| **FalkorDB / FalkorDBLite** | **Lowest**: small container, or embedded Python via FalkorDBLite | Medium: likely custom writer/reader adapters and backend-specific DDL | Partial/newer Bolt exists, but safest path is `falkordb-py` | **Best** for "easy local graph DB" | OpenCypher subset; validate exact queries/indexes | **Primary pick** |
| **ArcadeDB** | Medium: separate server/container for Python apps | **Lowest**: Neo4j driver over Bolt may mostly work | **Strong**: official Neo4j drivers over Bolt | Good, but not as frictionless as embedded Python | Java/server process; validate our Neo4j-specific queries | Compatibility fallback / first smoke test if implementation speed dominates |
| **Memgraph** | Medium: separate server/container | Low-medium: Bolt-native, but DDL/query differences remain | Strong: Neo4j drivers generally work | Good | Does not win setup simplicity; no graphiti backend | Fallback if ArcadeDB/FalkorDB fail |
| **Apache AGE** | Low if Postgres already exists; medium if extension setup is painful | High: adapter rewrite through SQL + AGE agtype | No | Good for "one DB stack" | Not a Bolt/Cypher drop-in; extension/install complexity | Later exploration only |
| **Kuzu / forks** | Very low embedded shape | Medium-high | No | Would be good technically | Upstream archived Oct 2025; forks not proven | Do not adopt now |

**Proposed implementation direction:** add `GRAPH_DB_BACKEND=neo4j|falkordb`
with Neo4j as default/prod; implement FalkorDB adapters behind
`GraphWriterPort` and `ClaimQueryPort`; support both `FALKORDB_URL` for a
container and a FalkorDBLite mode for the easiest local path. Leave the
current Neo4j path untouched.

---

## 2026-05-28 — FalkorDB implementation plan

Decision confirmed: start with **FalkorDB** for the local context-graph
backend. Optimize for low operational overhead first; keep Neo4j as the
default production path.

### Design goal

The rest of context-engine should not know which graph backend is active.
Only settings + adapter wiring should branch:

- writes stay behind `GraphWriterPort`
- reads stay behind `ClaimQueryPort`
- `ContextGraphService`, `ReadOrchestrator`, readers, reconciliation, and
  HTTP/CLI surfaces should be unchanged

### Phase 0 — Compatibility spike

Purpose: prove the exact Position-B claim graph shape works in FalkorDB before
touching the runtime wiring.

Spike script/tasks:

1. Start FalkorDB with the normal server/container path:
   `docker run -p 6379:6379 falkordb/falkordb`.
2. In a scratch script or focused integration test, connect with
   `falkordb-py`.
3. Validate the exact primitives context-engine needs:
   - create / merge `(:Entity {group_id, entity_key})`
   - set additional labels (`Entity`, canonical entity labels)
   - create / merge `:RELATES_TO` edges with properties
   - query edge property maps equivalent to Neo4j's `r{.*}`
   - filter by `group_id`, `name`, subject/object keys, validity dates,
     source system, and labels
   - delete all nodes for one `group_id`
4. Decide whether FalkorDB's Bolt path is mature enough to reuse the Neo4j
   driver. Default assumption: use `falkordb-py` for correctness and explicit
   backend control.

Exit criteria: one write/read/reset round trip passes with the same logical
`ClaimRow` output as the in-memory/Neo4j-backed readers.

### Phase 1 — Settings + dependency surface

Add backend selection and FalkorDB-specific settings.

Proposed env — follow the existing `CONTEXT_ENGINE_`-prefix-with-bare-fallback
convention used by the Neo4j settings (`context_engine_neo4j_uri()` reads
`CONTEXT_ENGINE_NEO4J_URI` → falls back to `NEO4J_URI`). Mirror that so a
dedicated context-engine value can override a shared one:

- `GRAPH_DB_BACKEND=neo4j|falkordb` (default: `neo4j`)
- `CONTEXT_ENGINE_FALKORDB_URL` → fallback `FALKORDB_URL`
  (e.g. `redis://localhost:6379`) for server/container mode
- `CONTEXT_ENGINE_FALKORDB_GRAPH_NAME` → fallback `FALKORDB_GRAPH_NAME`
  (default `context_graph`)
- `CONTEXT_ENGINE_FALKORDB_MODE` → fallback `FALKORDB_MODE`
  (`server|lite`, default `server`; `lite` = embedded local)

Code changes:

- extend `ContextEngineSettingsPort` with the backend selector + FalkorDB
  accessors (`graph_db_backend()`, `falkordb_url()`, `falkordb_graph_name()`,
  `falkordb_mode()`)
- update `EnvContextEngineSettings` to read the env above with the same
  `CONTEXT_ENGINE_*` → bare fallback + `.strip()`-to-`None` pattern as the
  existing `neo4j_*` accessors
- keep existing `neo4j_uri/user/password` methods unchanged for compatibility
- add tests for env precedence/defaults (incl. `CONTEXT_ENGINE_*` overriding
  bare, and `GRAPH_DB_BACKEND` default = `neo4j`)

Dependency decision:

- PyPI package is **`FalkorDB`** (capitalised; import name `falkordb`) — pin
  that, not a guessed `falkordb-py`.
- Add it as an **optional extra** (`[project.optional-dependencies] falkordb`),
  *not* a base dependency — mirrors the existing `graph` / `github` extras so
  the default install stays lean and prod (Neo4j) is unaffected.
- `falkordblite` (embedded) goes in the same `falkordb` extra (or a separate
  `falkordb-lite` extra) and is wired only when `FALKORDB_MODE=lite`; defer
  actually enabling Lite to PR 2 (see first-PR scope).

### Phase 2 — FalkorDB writer adapter

Create `adapters/outbound/graph/falkordb_writer.py` implementing
`GraphWriterPort`.

Members to implement (the full port surface — do not skip `enabled`):

- **`enabled` property** — *required; the whole service gates on it.*
  `ContextGraphService.enabled` returns `graph_writer.enabled`, and every
  writer method early-returns on `not self.enabled`. Compute it from
  **FalkorDB** config, not Neo4j creds: `settings.is_enabled()` **and**
  FalkorDB reachable/configured (`falkordb_url()` set, or `mode == 'lite'`).
  Mirror the structure of `Neo4jGraphWriter.enabled`
  ([`neo4j_writer.py:79`](app/src/context-engine/adapters/outbound/graph/neo4j_writer.py#L79))
  but swap the credential check. If omitted, the service silently reports
  disabled and all writes become no-ops.
- `ensure_indexes` (returns `bool`)
- `upsert_entities`
- `upsert_edges`
- `delete_edges`
- `invalidate`
- `reset_pot`

Implementation notes:

- generate UUIDs and timestamps client-side (`uuid4`, current UTC/time millis)
  instead of relying on `randomUUID()` / `timestamp()`
- avoid Neo4j-only `CALL ... IN TRANSACTIONS`; implement reset as a
  client-side loop scoped to `group_id`
- **`reset_pot` must return the same result-dict contract** as the Neo4j
  writer, because `ContextGraphService.reset_pot` wraps it and reads
  `inner.get("ok")` then surfaces `group_id_nodes_before` /
  `group_id_nodes_remaining`
  ([`context_graph_service.py:181-198`](app/src/context-engine/adapters/outbound/graph/context_graph_service.py#L181-L198)).
  Keys to return: `ok: bool`, `group_id_nodes_before: int`,
  `group_id_nodes_remaining: int`, plus `error` on failure. Also call
  `_require_valid_pot_id(pot_id)` first and fail closed on a bad partition
  key, same as the Neo4j path.
- keep Position-B edge shape unchanged:
  `(:Entity {group_id, entity_key})-[:RELATES_TO {...}]->(:Entity)`
- preserve the same idempotency key for edges:
  `(group_id, name, subject_key, object_key, source_ref)`
- make index creation best-effort, backend-specific, and non-fatal where
  FalkorDB syntax/support differs from Neo4j

### Phase 3 — FalkorDB reader adapter

Create `adapters/outbound/graph/falkordb_reader.py` implementing
`ClaimQueryPort`.

Required behavior:

- return the same `ClaimRow` shape as `Neo4jClaimQueryStore`
- support all current filters in `_FIND_CLAIMS_CYPHER`
- keep Python token-overlap semantic scoring for `fact_query`
- implement `entity_labels`

Important portability point: if FalkorDB does not support Neo4j-style map
projection (`r{.*}`), normalize records from FalkorDB's returned property
maps in Python.

### Phase 4 — Container wiring

Update `build_container` ([`container.py:214`](app/src/context-engine/bootstrap/container.py#L214))
so backend selection happens once, at the two lines that currently name Neo4j
([`:245`](app/src/context-engine/bootstrap/container.py#L245) writer,
[`:249`](app/src/context-engine/bootstrap/container.py#L249) claim store):

- `neo4j` → existing `Neo4jGraphWriter` + `Neo4jClaimQueryStore`
- `falkordb` → `FalkorDBGraphWriter` + `FalkorDBClaimQueryStore`

Switch **both together** (the writer feeds `ContextGraphService`; the claim
store feeds `ReadOrchestrator`). `ContextGraphService`, `ReadOrchestrator`,
`apply_reconciliation_plan`, and the container's exposed `graph_writer` field
all stay as-is — they only depend on the ports.

One change point is enough: `build_container_with_github_token` **delegates**
to `build_container` ([`container.py:470`](app/src/context-engine/bootstrap/container.py#L470)),
so both builders pick up the selection automatically — no duplication.

Keep default behavior unchanged. Production remains Neo4j unless explicitly
configured otherwise.

### Phase 5 — Tests

Unit tests:

- settings/defaults (incl. `CONTEXT_ENGINE_*` overriding bare env, and
  `GRAPH_DB_BACKEND` default = `neo4j`)
- container selects the correct writer/query store per `GRAPH_DB_BACKEND`
- **`FalkorDBGraphWriter.enabled`** reflects FalkorDB config (true when
  enabled + configured; false when `CONTEXT_GRAPH_ENABLED` off or FalkorDB
  unconfigured) — guards the silent-no-op gap
- **`reset_pot` return shape** matches the Neo4j contract (`ok`,
  `group_id_nodes_before`, `group_id_nodes_remaining`; `error` on failure)
  and rejects an invalid `pot_id`
- FalkorDB row normalization and filter parameter building
- writer helper functions for UUID/timestamp/idempotency props

Integration tests:

- mark live FalkorDB tests separately, similar to existing Neo4j integration
  tests
- cover:
  - entity + edge upsert
  - duplicate write idempotency
  - invalidation
  - `find_claims` filters
  - `entity_labels`
  - `reset_pot`

Parity test idea:

- run the same small Position-B fixture through `InMemoryClaimQueryStore`,
  Neo4j, and FalkorDB; compare `ClaimRow` results after normalizing UUID/time.

### Phase 6 — Local-dev documentation

Document two supported local modes:

1. Container mode:
   - run FalkorDB container
   - set `GRAPH_DB_BACKEND=falkordb`
   - set `FALKORDB_URL=redis://localhost:6379`
2. Lite mode:
   - install the optional FalkorDBLite dependency
   - set `GRAPH_DB_BACKEND=falkordb`
   - set `FALKORDB_MODE=lite`

Add a short troubleshooting section for:

- missing dependency
- FalkorDB not reachable
- unsupported Cypher/index syntax
- falling back to Neo4j

### Recommended first PR scope

Keep the first PR small and reversible:

1. Add settings + container backend selection.
2. Add FalkorDB reader/writer adapters.
3. Add server/container-mode integration tests.
4. Document FalkorDB container setup.

Defer FalkorDBLite to a second PR unless the server-mode adapter is already
passing cleanly. That keeps the first implementation focused on graph
semantics before adding embedded lifecycle management.

### Open validation questions

- ~~Exact FalkorDB client package/import name~~ — resolved: PyPI **`FalkorDB`**,
  import `falkordb`. Version pin still TBD (pick a current release in Phase 0).
- Whether FalkorDB supports all needed `MERGE` patterns with relationship
  properties exactly as written, or whether upserts need a MATCH-then-CREATE
  pattern.
- Whether label setting/querying matches Neo4j closely enough for
  `subject_label` / `object_label` filters.
- Best index syntax for:
  - `Entity(group_id, entity_key)`
  - `RELATES_TO(group_id, name)`
  - temporal claim filters
- Whether relationship vector indexes are usable later; not required for the
  first adapter because `fact_query` already uses Python scoring.

---

## 2026-05-28 — Plan review against the code (gaps fixed)

Validated the FalkorDB plan against the actual context-engine source. Names,
seams, and the "ContextGraphService unchanged" assumption all checked out
(`build_container`, `EnvContextEngineSettings`, `GraphWriterPort`,
`ClaimQueryPort` = `find_claims`+`entity_labels`, the `:245`/`:249` swap
point). `ContextGraphService` / `ReadOrchestrator` / `apply_reconciliation_plan`
are genuinely backend-agnostic. Four gaps found and folded into the phases
above:

1. **`enabled` was missing from the writer surface.** It gates the whole
   service (`ContextGraphService.enabled` → `graph_writer.enabled`; every
   writer method early-returns on it). Must be computed from FalkorDB config,
   not Neo4j creds. → added to Phase 2 as a required member + a Phase 5 test.
2. **`reset_pot` return contract.** The service reads `ok` /
   `group_id_nodes_before` / `group_id_nodes_remaining` off the writer's
   result; the FalkorDB reset must return that exact dict. → documented in
   Phase 2 + a Phase 5 shape test.
3. **Dependency packaging.** Pin PyPI `FalkorDB` (import `falkordb`), add as an
   optional extra (mirrors `graph`/`github`), not a base dep. → Phase 1 +
   open-questions resolved.
4. **Env naming.** Follow the existing `CONTEXT_ENGINE_*` → bare-fallback
   convention. → Phase 1 env block updated.

Also confirmed `build_container_with_github_token` delegates to
`build_container`, so the single Phase-4 change covers both builders.

**Status:** Plan reviewed and corrected; ready to implement Phase 0 spike on
request. Still no code written.

---

## 2026-05-28 — Linear issues created (tracking)

The plan is now tracked in Linear. Parent epic + seven sub-issues mapping
1:1 to the phases above. Implementation deferred — we pick these up as we
progress.

**Parent:** [POT-1420 — New graph adapter for local-first](https://linear.app/potpie/issue/POT-1420/new-graph-adapter-for-local-first)

**Sub-issues** (dependency-ordered; the spike gates everything):

1. [POT-1421 — FalkorDB compatibility spike (Position-B claim graph)](https://linear.app/potpie/issue/POT-1421/falkordb-compatibility-spike-position-b-claim-graph)
   — Phase 0. Prove the claim-graph shape on FalkorDB before any production
   wiring. **Gate for all others.**
2. [POT-1422 — Backend selector + FalkorDB settings/dependency](https://linear.app/potpie/issue/POT-1422/backend-selector-falkordb-settingsdependency)
   — Phase 1. `GRAPH_DB_BACKEND` selector + FalkorDB env/settings + optional
   dependency; Neo4j stays default.
3. [POT-1423 — FalkorDBGraphWriter (GraphWriterPort)](https://linear.app/potpie/issue/POT-1423/falkordbgraphwriter-graphwriterport)
   — Phase 2. Writer adapter with matching write semantics, `reset_pot`
   contract, and `enabled`-gating.
4. [POT-1424 — FalkorDBClaimQueryStore (ClaimQueryPort)](https://linear.app/potpie/issue/POT-1424/falkordbclaimquerystore-claimqueryport)
   — Phase 3. Reader/query adapter with parity on row shape, filters,
   normalization.
5. [POT-1425 — Wire graph-backend selection into build_container](https://linear.app/potpie/issue/POT-1425/wire-graph-backend-selection-into-build-container)
   — Phase 4. Switch writer + claim store together from config.
6. [POT-1426 — FalkorDB integration + parity tests](https://linear.app/potpie/issue/POT-1426/falkordb-integration-parity-tests)
   — Phase 5. Live FalkorDB integration tests + parity vs in-memory & Neo4j.
7. [POT-1427 — Document FalkorDB local-dev modes](https://linear.app/potpie/issue/POT-1427/document-falkordb-local-dev-modes)
   — Phase 6. Container + Lite developer workflows.

Staging: validate feasibility (1) → config surface (2) → writer (3) + reader
(4) → container wiring (5) → parity tests (6) + docs (7). First PR =
2+3+4+5 + container-mode tests; FalkorDBLite (Lite mode) deferred to a second
PR.

**Status:** Tracked in Linear, not yet started. Resume at POT-1421 (spike)
when implementation begins.

---

## 2026-05-29 — Phase 0 spike EXECUTED against live FalkorDB (POT-1421)

Ran a real compatibility spike, not a paper one. Brought up
`falkordb/falkordb:latest` (image `1.6.x`) on port 6399 via Docker, installed
the PyPI **`falkordb`** client (resolved **1.6.1**, pulls `redis` 8.x) into the
context-engine `.venv`, and probed every exact Position-B primitive the Neo4j
adapters use. Two throwaway scripts (`.tmp-falkordb-spike*.py`, deleted after).

### What PASSED (so it can be reused verbatim)

- `randomUUID()` and `timestamp()` — **both supported** (the plan assumed they
  weren't). So the `ON CREATE SET r.uuid = randomUUID(), r.created_at =
  timestamp()` writes in `cypher.py` work unchanged — no client-side UUID/clock
  needed.
- `MERGE (:Entity {group_id, entity_key}) ON CREATE SET ... SET e += $props`.
- Dynamic second label `SET e:Service` on an existing node, and `labels(e)`
  returns `['Entity','Service']` — so canonical-label setting + the
  `subject_label`/`object_label` filters work.
- `MERGE (a)-[r:RELATES_TO {group_id,name,subject_key,object_key,source_ref}]->(b)`
  with `ON CREATE SET` + `SET r += $props`; **idempotent** on re-MERGE (count
  stays 1).
- `r{.*}` map projection (returns an `OrderedDict`, which `dict(...)` in
  `_row_from_record` handles fine).
- The **entire `_FIND_CLAIMS_CYPHER`** shape: `IN` lists, `IS NULL OR ...`
  guards, `as_of`/`va_after`/`va_before` temporal predicates, and the
  `labels(a)`/`labels(b)` label filters — all return correct rows.
- `entity_labels` query; `count(n)` aggregates; param maps with `None` values;
  `FalkorDB.from_url("redis://host:port")` + `select_graph(name)`.
- **Composite indexes** in the *unnamed* form:
  `CREATE INDEX FOR (n:Entity) ON (n.group_id, n.entity_key)` and
  `CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id, r.name)`.

### What FAILED → gaps + fixes folded into the implementation

1. **Named indexes + `IF NOT EXISTS` are NOT supported.** FalkorDB only accepts
   `CREATE INDEX FOR (n:L) ON (n.p)` (no index name, no `IF NOT EXISTS`). So
   `cypher.py:ensure_canonical_indexes` (named + `IF NOT EXISTS`) can't be
   reused. → **Fix:** FalkorDB adapter ships its own *unnamed* index DDL,
   each statement wrapped in try/except so a re-run "already indexed" error is
   swallowed (best-effort, non-fatal — matches the plan's index stance).
2. **Relationship VECTOR index uses different syntax** (`CREATE VECTOR INDEX`
   Neo4j form is rejected; FalkorDB wants `OPTIONS {dimension:…,
   similarityFunction:…}`). Already non-critical — `fact_query` runs the Python
   token-overlap fallback. → **Fix:** skip the vector index for now; revisit if
   native vector search is ever put on the hot path.
3. **`CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF $batch ROWS` is NOT
   supported** (confirmed). → **Fix:** `reset_pot` uses a client-side batched
   loop (`MATCH (n {group_id}) WITH n LIMIT $n DETACH DELETE n` until 0), which
   the spike verified works.

### Architecture decision (revises the plan)

Because the mutation Cypher (`randomUUID`/`timestamp`/`MERGE`/dynamic labels)
all work, **the FalkorDB writer reuses `cypher.py`'s async functions
(`upsert_entities_async`, `upsert_edges_async`, `delete_edges_async`,
`apply_invalidations_async`) through a thin async-driver shim** over the sync
`falkordb` client — instead of duplicating ~450 lines of bitemporal /
supersession logic. The reader reuses `neo4j_reader`'s `_FIND_CLAIMS_CYPHER`,
`_ENTITY_LABELS_CYPHER`, `_row_from_record`, `_iso`, `_embedding_score`.
Only three things are FalkorDB-specific: the driver shim, the unnamed-index
DDL in `ensure_indexes`, and the client-side `reset_pot` loop. This keeps the
two backends semantically identical and the new code small (Simplicity First).

`result.header` shape is `[[type_code, name], ...]`; `result_set` is a list of
lists — the shim maps each row to `dict(zip(col_names, row))` so reused
Neo4j-shaped record access (`rec["props"]`, `rec["cnt"]`, `rec["labels"]`)
works untouched.

**Status:** Phase 0 (POT-1421) complete and GREEN. Proceeding to implement
Phases 1–4 (settings + writer + reader + container wiring) + Phase 5 tests in
one PR; FalkorDBLite deferred.

---

## 2026-05-29 — Phases 1–6 IMPLEMENTED + tested (POT-1422…1427)

First PR landed end to end. 27 new unit tests + 1 live integration round-trip
all green; the full context-engine unit suite is unchanged (875 pass; the 10
failures are pre-existing and unrelated — missing `tests/data/linear/*.json`
fixtures, agent-bundle file creation, benchmarks smoke).

### What shipped

**Phase 1 — settings + dependency** (POT-1422)
- `domain/ports/settings.py`: added `graph_db_backend()`, `falkordb_url()`,
  `falkordb_graph_name()`, `falkordb_mode()` to `ContextEngineSettingsPort`
  **with concrete default bodies** (backend `neo4j`, url `None`, name
  `context_graph`, mode `server`) — see "gap 1" below.
- `adapters/outbound/settings_env.py` (`EnvContextEngineSettings`) **and**
  `app/modules/context_graph/wiring.py` (`PotpieContextEngineSettings`):
  implemented all four with the `CONTEXT_ENGINE_*` → bare-fallback `.strip()`
  convention. Both were needed — see "gap 2".
- `app/src/context-engine/pyproject.toml`: `falkordb = ["falkordb>=1.6.1"]`
  optional extra, **not** in `all` (keeps the monolith's `context-engine[all]`
  dependency lean — see "gap 3").

**Phase 2 — writer** (POT-1423) `adapters/outbound/graph/falkordb_writer.py`
- `_FalkorAsyncDriver`/`_FalkorAsyncSession`/`_FalkorAsyncResult`: a ~40-line
  async shim over the sync `falkordb` client exposing the exact slice of the
  Neo4j async-driver API that `cypher.py` uses (`async with`, `await run`,
  `await single`/`await consume`). This lets the writer **reuse** cypher.py's
  `upsert_entities_async` / `upsert_edges_async` / `delete_edges_async` /
  `apply_invalidations_async` verbatim.
- FalkorDB-specific bits only: `enabled` (gated on FalkorDB config, not Neo4j
  creds), unnamed best-effort `ensure_indexes`, and a client-side batched
  `reset_pot` returning the exact `{ok, group_id_nodes_before,
  group_id_nodes_remaining}` contract.

**Phase 3 — reader** (POT-1424) `adapters/outbound/graph/falkordb_reader.py`
- Reuses `neo4j_reader`'s `_FIND_CLAIMS_CYPHER`, `_ENTITY_LABELS_CYPHER`,
  `_row_from_record`, `_iso`, `_embedding_score`; only swaps the driver call +
  `result_set`→record mapping. fact_query keeps Python token-overlap scoring.

**Phase 4 — container** (POT-1425) `bootstrap/container.py`
- One branch in `build_container`: `GRAPH_DB_BACKEND=falkordb` swaps **both**
  writer + claim store to FalkorDB; default stays Neo4j. FalkorDB adapters
  import lazily. `build_container_with_github_token` delegates here, so both
  builders get it for free. `graph/__init__.py` documents the lazy import.

**Phase 5 — tests** (POT-1426)
- `test_settings_graph_backend.py` (env precedence/defaults),
  `test_falkordb_writer.py` (enabled gate ×4, reset_pot shape + invalid pot_id
  + disabled, best-effort/unnamed index DDL, shim record mapping, reused-MERGE
  path), `test_falkordb_reader.py` (params/parsing/fact_query/limit/labels,
  mirrors `test_neo4j_claim_query`), `test_container_backend_selection.py`
  (real `build_container` selection both ways), and
  `tests/integration/test_falkordb_roundtrip.py` (live write→read→reset +
  idempotency; skips unless `FALKORDB_TEST_URL` is reachable).

**Phase 6 — docs** (POT-1427)
- `compose.yaml`: opt-in `falkordb` service under `profiles: ["falkordb"]` on
  host port **6399** (avoids the redis broker on 6379). `.env.template`: backend
  selection block. `docs/context-graph/README.md`: local-dev + troubleshooting.

### Gaps found *while coding* and how they were fixed

1. **The settings interface is a `Protocol` with two real implementers.**
   Adding bare `...` stubs would have forced edits to every implementer and any
   test fake. → Gave the four new methods **concrete default bodies** in the
   Protocol; classes that subclass it inherit a safe `neo4j` default. Only the
   two real env-backed settings override them.
2. **There are *two* settings classes, not one.** `EnvContextEngineSettings`
   (context-engine default) and `PotpieContextEngineSettings` (the monolith /
   `build_container_for_session` path that an individual user actually runs).
   The plan only named the first. → Implemented the accessors in **both**, or
   FalkorDB would never activate for the real local user.
3. **`falkordb` in the `all` extra would become a hard dep.** Root
   `pyproject.toml` depends on `context-engine[all]`, so anything in `all`
   ships to every monolith install. → Kept `falkordb` as a standalone extra,
   imported lazily in the adapters + container branch.
4. **`enabled` for `lite` mode would be a silent-no-op trap.** The plan had
   `enabled` true for `mode == 'lite'`, but Lite isn't wired this PR (no URL →
   writes would error or no-op confusingly). → Scoped `enabled` to server mode
   (URL set, or an injected graph for tests); `lite` stays reserved/off until
   PR 2. Documented in code + README.
5. **`ReadOrchestrator.claim_query` is a public field, not `_claim_query`.**
   First draft of the container test asserted the wrong attribute. → Fixed.

(Phase-0 also already corrected the plan's assumptions that `randomUUID()` /
`timestamp()` were unsupported, and that named/`IF NOT EXISTS` index DDL would
work — see the Phase-0 section above.)

### Verification

- New unit tests: **27 passed**.
- Live integration round-trip vs `falkordb/falkordb:latest` (port 6399):
  **passed** — entity+edge upsert, edge idempotency, `find_claims`,
  `entity_labels`, and `reset_pot` (before=N → remaining=0) all behave like the
  Neo4j path.
- No new lint errors. Default (Neo4j) path untouched and still selected unless
  `GRAPH_DB_BACKEND=falkordb`.

**Status:** First PR (Phases 1–6, server/container mode) complete and GREEN.
Deferred to a follow-up: FalkorDBLite embedded mode (`FALKORDB_MODE=lite`) and
the optional native relationship vector index.
