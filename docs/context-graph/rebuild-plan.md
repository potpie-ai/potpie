# Context Engine — Rebuild Plan (v4)

> **Status:** active execution plan, v4 (2026-05-20). **Position B is
> baked in** (substrate POC). v4 folds in the **proper POC** findings —
> an end-to-end ingestion + read benchmark across all four UCs against
> real LLM extraction — which surfaced ingestion-quality bottlenecks
> that change phase ordering and scope. Supersedes [`plan.md`](./plan.md)
> and v2 entirely.
>
> **Inputs (read these first):**
> - [`vision.md`](./vision.md) — what we're building and why.
> - [`design-review.md`](./design-review.md) — the diagnosis. The
>   problem sets (PS1–PS8) and the resolved decisions (D0–D3, E1–E4) this
>   plan operationalizes. Read the executive summary at minimum.
> - [`bench-plan.md`](./bench-plan.md) — the benchmark taxonomy
>   (PREF / INFRA / TIME / BUG + COMBO) this plan is validated against.
> - **[`../../pocs/position_b/`](../../pocs/position_b/)** — substrate
>   POC. 9 tests, all green. Validates the canonical edge shape this plan
>   adopts in P0. Read its `README.md` before P0/P2.
> - **[`../../pocs/position_b_proper/`](../../pocs/position_b_proper/)** —
>   end-to-end POC: real LLM extraction + identity layer + 4 scenarios
>   (one per UC) + bench-style scoring against distractors. Read its
>   `findings.md` before P2/P3/P4/P5. Headline: **39.6% coverage average,
>   83.3% precision**, with specific failure modes the v4 phases address.
>
> **The four use cases this plan must serve concretely:**
> 1. **PREF** — project preferences (coding patterns, libs, error handling,
>    logging, structure). Surface to agent when writing code.
> 2. **INFRA** — project infra topology (services, environments,
>    adapter-per-env, deployments, dependencies). Surface to agent when
>    debugging/changing infra.
> 3. **TIME** — timeline of recent activity (PRs, tickets, deploys, alerts)
>    across sources. Surface to agent when investigating cause-over-time.
> 4. **BUG** — durable problem→solution memory (incl. dev-setup gotchas).
>    Surface to agent when stuck on a symptom.
>
> More use cases will be added. The plan is built around **primitives** the
> four share, not features specific to them, so future use cases compose
> from the same parts without re-architecting.

---

## Position B — the canonical edge shape (adopted, POC-validated)

Every claim in the canonical graph is one **`:RELATES_TO` edge** between
entities keyed by our deterministic `(group_id, entity_key)` identity. The
edge carries the predicate, bitemporal validity, full provenance, and the
human-readable fact text for embedding:

```cypher
MERGE (a)-[r:RELATES_TO {
    group_id: $pot,
    name: $predicate,           -- e.g. "DEPENDS_ON", "OWNED_BY", "POLICY_APPLIES_TO"
    subject_key: $subject_key,
    object_key: $object_key,
    source_ref: $source_ref     -- distinguishes corroborating claims
}]->(b)
ON CREATE SET
    r.uuid = randomUUID(),
    r.created_at = $now,
    r.expired_at = null,
    r.invalid_at = null
SET
    r.valid_at = $valid_at,         -- event time
    r.source_system = $source,
    r.evidence_strength = $strength,
    r.fact = $fact,                 -- text for embedding + agent reading
    r.confidence = $confidence,     -- optional; usually derived
    r.observed_at = $now
```

The MERGE key includes `source_ref` so two sources making the same claim
produce two edges; re-scanning the same source updates the existing edge
idempotently in place. Bitemporal point-in-time = one Cypher predicate:

```cypher
WHERE (r.valid_at  IS NULL OR r.valid_at  <= $as_of)
  AND (r.invalid_at IS NULL OR r.invalid_at > $as_of)
```

Supersession (newer same-(subject, predicate) claim with different object)
is five lines of Cypher. Semantic similarity over claim facts uses native
**Neo4j 5.x relationship vector indexes** (`db.index.vector.queryRelationships`
on `r.fact_embedding`). Belief derivation aggregates live claims by
strength × recency × corroboration into a coarse `{high|medium|low|unknown}`
label — 30 lines of Python.

**What the POC proved (9 tests, all green):** deterministic identity
preserved, edge writes work directly via `g.driver`, two-source
corroboration aggregates into one belief, Cypher and Graphiti's
`SearchFilters` DSL express identical point-in-time predicates, supersession
flips beliefs and preserves history, blast-radius traversal works at POC
scale, and Neo4j's native relationship vector index returns the right
fact for symptom-shaped queries. See `pocs/position_b/README.md` for the
verdicts and `pocs/position_b/poc.py` for runnable evidence.

**What this changes in the plan vs the v2 framing:**
- **P0 includes a one-shot migration** of existing typed edges
  (`:OWNS`, `:DEPENDS_ON`, …) to `:RELATES_TO {name: <type>}`. There is no
  "port the temporal vocabulary" work — the temporal vocabulary lives on
  these edges from the start.
- **P2 (claim model) shrinks substantially.** Storage representation,
  MERGE-key semantics, supersession mechanics, and the belief-derivation
  shape are settled by the substrate POC. P2 reduces to the belief
  refinements (per-source weighting, decay, equal-recency conflict
  surfacing) — about half its prior scope.
- **P7 (ranking) gets semantic similarity for free.** Native Neo4j vector
  query supplies the similarity factor; the ranker is a thin weighting
  layer over five inputs.
- **Graphiti's role narrows further.** Post-POC, Graphiti contributes only
  the Neo4j driver wrapper and `add_episode` for audit-log persistence.
  Native Neo4j supersedes its embedding/vector path. Whether to keep
  Graphiti at all becomes a P0 sub-decision with revisit at P5.

---

## Proper POC findings — the LLM extraction bottleneck (v4 input)

The substrate POC validated *mechanics*. The proper POC validated the
*ingestion + read path end-to-end* against an Acme-shaped fixture (15
signal events + 7 distractors + 3 universe seeds, multi-source, real LLM
extraction via `gpt-5.4-mini`, native vector indexes, bench-style
scoring). Headline scorecard:

| Dim   | Coverage | Precision | Phrases | Hallucinations | Conf-match |
|-------|---------:|----------:|--------:|---------------:|-----------:|
| PREF  |    33.3% |     66.7% |   33.3% |        **yes** |        100% |
| INFRA |     0.0% |    100.0% |    0.0% |             no |        100% |
| TIME  |    25.0% |    100.0% |    0.0% |             no |        100% |
| BUG   |   100.0% |     66.7% |  100.0% |             no |        100% |
| **Avg** | **39.6%** | **83.3%** |        |                |            |

**The substrate worked. The ingestion path didn't fully cooperate.** Five
specific failure modes surfaced — each maps to a phase change in v4.

### F1 — INFRA 0% coverage. Service ↔ Deployment ontology gap

The LLM extractor classified `auth-svc` from k8s manifests as a
`Deployment` entity (key `deploy:auth-svc`), not a `Service`. So claims
`-[USES]-> postgres-auth-prod` attach to the Deployment, not the Service.
The INFRA reader queried the Service entity and found nothing about its
dependencies.

**v4 change:** P3 makes the **Service ↔ Deployment ontology join** a
named, required refinement (was implicit). INFRA reader (P9) must hop
`Service → Deployment → USES → DataStore`.

### F2 — CODEOWNERS produced `(component:unknown)`. Deterministic activity layer is more urgent than v3 framed it

LLM extractor over the CODEOWNERS body saw `*` (wildcard) and reasonably
emitted `(component:unknown) -[OWNED_BY]-> (person:alice)`. The
contextual fact — the file lives at `apps/auth/CODEOWNERS` — was lost
because the extractor only sees body text.

**v4 change:** P5's deterministic activity layer is **promoted to
parallel with P4** rather than after. The CODEOWNERS scanner needs to
stamp scope deterministically from the file path *before* handing body
text to the LLM. Same for k8s manifests at `clusters/<env>/<service>.yaml`.
P4 source-scanning depends on P5 path-aware-scope-stamping for quality.

### F3 — PREF leaked superseded ADR-003. Edge-contradiction detection is load-bearing

ADR-007 explicitly supersedes ADR-003. The LLM extracted both as
policies but emitted **no `SUPERSEDES` edge** between them. The PREF
reader's superseded-subject filter had nothing to filter on. ADR-003
surfaced alongside ADR-007.

**v4 change:** P2 adds **edge-contradiction detection as first-class**
(was implicit/assumed). Two paths:
- **(preferred for V1) Port Graphiti's `dedupe_edges.resolve_edge`
  pipeline** (~200 LoC per the Graphiti research, `edge_operations.py:
  622-824`). LLM-driven contradiction check per new edge; the only
  mechanism that emits `SUPERSEDES` when source text doesn't say it.
- **(complementary) Deterministic ontology rules:** Policy with same
  `policy_kind + code_scope` is implicitly singleton; new one supersedes
  old. Handles structured cases; the LLM path handles unstructured.

### F4 — TIME 25%. PR-merged events don't link to scope. MENTIONS-style provenance needed

LLM emitted `(person:alice) -[MERGED_BY]-> (pr:1042)` for PR-merged
events. Neither endpoint is `service:auth-svc`. The TIME reader's "find
claims where service appears as subject or object" filter missed PR
events entirely.

**v4 change:** P5 adds **`MENTIONS`-style episode-to-entity linking** to
the activity layer (Graphiti research flagged this explicitly: "If you
skip MENTIONS, you lose provenance. Port it"). Every entity mentioned in
an episode body gets a `:RELATES_TO {name: 'MENTIONED_IN'}` claim to the
episode. TIME reader queries activities whose episode mentions the
scope entity. Our existing `episode_uuid` property on claims gives part
of this; we extend to MENTIONS edges proper.

### F5 — Confidence label `high` on 0% coverage. Coverage-gap detection missing from confidence derivation

INFRA scenario returned `confidence=high` despite 0% recall. The
derivation reflects what we returned (strength + corroboration count of
the hits), not what we *should* have returned but didn't. PS3 in
miniature.

**v4 change:** P2 adds **coverage-gap detection** to the belief layer.
When a reader's expected scope yields claims but the answer's
*per-fact* confidence is uniformly low, OR when the planner expected
multiple evidence families and ≥1 returned empty, the envelope's
top-level confidence is capped at `medium` and `fallbacks` gets a typed
`coverage_gap` entry. The envelope cannot return `high` on a sparse
result.

### What stayed strong

- **Identity convergence worked** when the LLM emitted variants (`auth
  service` resolved to `service:auth-svc`; `Inventory Service` resolved
  to `service:inventory-svc`). P1's identity layer is sound; the
  bottleneck is what the LLM emits, not what the identity layer does
  with it.
- **BUG semantic retrieval is the strong UC** — 100% recall via native
  Neo4j vector index on `fact_embedding`. P9's PriorBugsReader sketch is
  validated.
- **Substrate primitives at scale work.** Writes, reads, traversal,
  vector queries, bitemporal `as_of` — every primitive ran correctly
  through realistic ingestion. Substrate is not the bottleneck.

### Implication for the bench-as-gating-signal mechanic

The proper POC's **39.6% baseline** is the starting line. v4 makes per-
phase bench coverage targets explicit:

| After phase | Bench coverage target (avg across PREF/INFRA/TIME/BUG) |
|---|---:|
| P0 (substrate migration only — no new ingestion) | ≥ 40% (no regression vs proper POC baseline) |
| P3 (Service ↔ Deployment join + ontology refinement) | ≥ 55% |
| P4 + P5 (deterministic scanning + activity layer) | ≥ 70% |
| P6 (recording UX) | ≥ 75% |
| P7 (ranking) | ≥ 80% |
| P10 (full bench validation) | bench target locked in §3 of bench-plan.md |

Targets are aspirational; the first phase to ship with a real bench-
coverage number sets the baseline, subsequent phases must monotonically
improve. A phase that doesn't move its target dimension up is the
signal to stop, write down what changed, replan.

---

## How to use this plan

- Phases are **sequential** within the load-bearing pillars (P0–P3) and
  **partially parallelizable** beyond (P4–P9). The capability rollup at the
  end shows the dependency graph explicitly.
- **Every phase starts with discovery.** Re-validate against current code
  before committing to specifics. The plan describes direction; specifics
  are decided in the phase that owns them.
- **No backwards-compatibility paths.** When a thing is replaced, the old
  thing goes in the same change. The cost of breakage is low; the cost of
  a confused codebase is high. One-shot data migrations are fine; runtime
  compat shims are not.
- **Docs and benchmark scenarios are part of every phase, not afterthoughts.**
  Each phase ends by reconciling `design-review.md`, `architecture.md`, and
  `bench-plan.md` with what was built and adding the bench scenarios that
  prove the phase's exit criteria.
- **If discovery reveals the direction was wrong: stop, write down what
  changed, update this plan, replan.** This is the living plan; rewrite as
  understanding deepens.

---

## Anti-goals (every phase)

- No second graph store, no parallel graph abstraction, no shadow subgraph.
  One Neo4j, one canonical graph.
- **The canonical edge shape is `:RELATES_TO {name, valid_at, invalid_at,
  ...}`** — no parallel typed-edge writes, no parallel temporal vocabulary.
  Deterministic identity is preserved on entities; edges are uniform.
- No new public agent tools beyond `context_resolve`, `context_search`,
  `context_record`, `context_status`. New capabilities become parameters.
- No re-introduction of deterministic *event-driven* plan compilers
  (the dead `propose_plan` path). Event sources go through the LLM agent.
- No source-specific code in `application/` or `domain/`. All source code
  lives in connectors (events) or scanners (config files).
- No full source payloads in the graph. References + compact summaries.
  Bodies fetch on demand.
- **No "Graphiti owns the entity model" framing.** Post-POC, Graphiti's
  contribution narrows to a Neo4j driver wrapper and `add_episode` for
  audit-log persistence; the Potpie reconciliation agent is the only LLM
  extractor that produces canonical entities; semantic search uses native
  Neo4j vector indexes.
- **No "Graphiti removal is cheap" assumption.** The proper POC plus the
  Graphiti-internals research show ~1,300 LoC to port equivalent
  behavior, with **~250 LoC of that being load-bearing edge-contradiction
  detection (`dedupe_edges.resolve_edge` + MENTIONS + temporal
  arithmetic)** that emits `SUPERSEDES` when source text doesn't say it.
  Without porting this, removal regresses ingestion quality on PREF and
  COMBO scenarios. **Removal target: 3-4 weeks of focused work** including
  bench-validation, not 2. Don't commit until F3-driven implementation
  in P2 hits its bench target.
- No `confidence: float` invented at the envelope layer. Confidence is
  derived from claim state or it doesn't exist.
- No phantom vocabulary. If `include=X` is accepted, X must produce a real
  reader response (or a typed `unsupported_include` fallback).
- No cross-pot federation. One pot, one tenant.

---

## Capability target (one paragraph)

By the end of this plan: every fact in the graph is a **claim** stored as
a `:RELATES_TO` edge between deterministically-identified entities,
carrying source, event-time (`valid_at`), observation-time (`created_at`),
`evidence_strength`, and a `fact` text representation embedded for
semantic search. Confidence is derived from corroboration × recency ×
scope × verification × **coverage-gap** — never a stored constant, and
capped when underlying recall is low (F5). Cross-source identity is
resolved through an inspectable alias layer. Four ingestion paths feed
the claim store: **source-scanning** (config files → high-strength
deterministic claims, with path-aware scope stamping from P5), **event-
driven with a deterministic activity layer + MENTIONS provenance**
(webhooks → activity facts + LLM-enriched bodies + episode-to-entity
mentions), **structured agent recording** (`context_record` per
record_type), and **inferential extraction** (low-strength derived
claims). An **edge-contradiction-detection step** in P2 emits
`SUPERSEDES` claims when source text doesn't explicitly say so —
addressing F3. Five intents and ~10 task-shaped include families drive
every read; the four-tool agent surface returns one envelope shape with
honest trust signals. Bitemporal queries work end-to-end via one Cypher
predicate; semantic similarity uses native Neo4j vector queries over
`r.fact_embedding`. The ontology carries the Service ↔ Deployment join
(F1) and other refinements that align with what the LLM extractor
actually produces. Graphiti's role is the Neo4j driver wrapper +
episode-write convenience; full removal is bench-validated, not
asserted. **Bench coverage averages ≥ 80% across PREF/INFRA/TIME/BUG**
(starting from 39.6% baseline; per-phase targets in the proper POC
findings section above).

---

## Sequencing rationale & phase list

The order is **load-bearing pillars first** (P0–P3), then **parallelizable
ingestion and retrieval** (P4–P9), then **validation** (P10). v4
restructures vs v3 based on the proper POC:
- **P2 grows** — adds edge-contradiction detection (F3) and coverage-gap
  confidence (F5).
- **P3 grows** — adds the Service ↔ Deployment ontology join (F1) as
  a named, required refinement.
- **P5 moves earlier, parallel with P4** — its path-aware scope stamping
  (F2) and MENTIONS provenance (F4) are prerequisites for P4's source
  scanners to deliver quality, not orthogonal additions.
- **P10 (bench) runs continuously from P4 onward** — the bench is the
  gating signal per the targets in the proper POC findings section, not
  a final-phase activity. Every phase from P3 onward must hit its
  bench-coverage target before declaring done.

| # | Phase | Pillar | Depends on | Bench target |
|---|---|---|---|---|
| 0 | Ground clearance + edge-shape migration | cleanup | — | ≥ 40% (no regression) |
| 1 | Identity contract | substrate | P0 | — |
| 2 | Claim model: belief + decay + contradiction + coverage-gap | substrate | P0, P1 | — |
| 3 | Ontology refinement (incl. Service ↔ Deployment) | substrate | P2 | ≥ 55% |
| 4 | Source-scanning ingestion (UC2) | ingestion | P3, **P5** | ≥ 70% |
| 5 | Event-driven deterministic activity layer + MENTIONS (UC3) | ingestion | P3 | ≥ 70% |
| 6 | Structured recording UX (UC1, UC4) | ingestion | P2, P3 | ≥ 75% |
| 7 | Ranking layer | retrieval | P2, P3 | ≥ 80% |
| 8 | One read surface, task-anchored | retrieval | P2, P3, P7 | — |
| 9 | Use-case readers | retrieval | P2–P8 | — |
| 10 | Continuous benchmark validation | validation | all (continuous from P3) | bench-plan locked |

P4 and P5 now have **explicit interdependence** — P5's path-aware scope
stamping is what makes P4's CODEOWNERS / k8s-manifest scanners produce
quality claims. P5 can ship without P4 (event-driven webhooks alone),
but P4 ships only after P5's scope-stamping is available. P6 can run in
parallel with P4/P5 after P3. P7 can start once P2 lands. P8 needs P7.
P9 ships per-UC as its ingestion lands.

---

## Phase 0 — Ground clearance + edge-shape migration

**Goal:** delete what's dead, demote what's misplaced, **migrate the
canonical graph from typed edges to `:RELATES_TO`** (Position B), and
unify the bitemporal vocabulary. End with a smaller codebase, a clear
write path, the POC's edge shape in production use, and the existing
supersession / conflict machinery re-pointed at the canonical graph.

**Why now.** Three years of accreted aspirational surface (`propose_plan`,
the parallel temporal vocabulary, two competing edge shapes) is actively
misleading every later phase. The POC validates the target shape. The
migration is mechanical Cypher; the deletions cost nothing because nothing
depends on them.

### Discovery

**POC follow-up tests** (do these first — they de-risk P2 and P4):

- Confirm `temporal_supersede.py` (`apply_predicate_family_auto_supersede`)
  runs unchanged against POC-shaped edges. It already matches
  `:RELATES_TO {group_id, name}` with `valid_at`/`invalid_at`; should be a
  no-op port. Run against a populated POC pot; verify older
  same-(subject, family) edges are stamped `invalid_at`.
- Confirm `family_conflict_detection.py` (`apply_family_conflict_detection`)
  runs unchanged. Construct two equal-time, equal-strength claims with
  different objects; verify a `QualityIssue {kind: 'conflict'}` node is
  created.
- Test multi-label entities: `SET n:Service:Activity` works via the
  dynamic-label pattern from the POC. Confirm the ontology-derived label
  inference still functions.
- Test environment-scoped claims: write `:RELATES_TO {name: 'DEPENDS_ON',
  environment: 'prod', ...}`; verify Cypher with `WHERE r.environment =
  $env` filters correctly and the composite index helps.
- Test re-scan idempotency: write the same claim twice with the same
  `source_ref`; confirm exactly one edge exists (MERGE updates in place).
- Smoke-test at scale: populate ~10k claim edges; measure blast-radius
  traversal (`:RELATES_TO*1..3` filtered by `name`) latency with the
  composite index. Verify <500ms; if not, tune the index (probably
  `(group_id, name, valid_at, invalid_at)`).

**Codebase audit:**

- `grep -ri "propose_plan\|build_.*_plan\|context_graph: object"` across
  `app/src/context-engine` and `docs/`. Confirm every hit is the port
  verb, its three connector implementations, the registry self-call, the
  test files, or doc language. No production caller.
- Inventory every site that writes typed edges today (`canonical_writer.py`
  edge upsert paths, `apply_plan.py`, the connector plan compilers being
  deleted anyway). These all need to write `:RELATES_TO` instead.
- Inventory the parallel temporal naming: `prov_valid_from`/`prov_valid_to`
  on canonical typed edges vs `valid_at`/`invalid_at` on Graphiti episodic
  edges. The migration unifies on `valid_at`/`invalid_at`.
- Check Graphiti's `add_episode` API in the installed `graphiti_core`:
  does any flag/parameter skip LLM extraction? The POC shows we don't
  need Graphiti's extraction or embeddings (native Neo4j vector works), so
  if extraction can be turned off, do that; otherwise ignore output.

### Decisions this phase

- **Adopt Position B as the canonical edge shape.** Settled by POC. All
  edge writes produce `:RELATES_TO` with the property set from
  `pocs/position_b/poc.py`.
- **Migration mechanic.** One-shot Cypher pass: for every existing
  `(:Entity)-[r:OWNS|DEPENDS_ON|...]->(:Entity)`, create the equivalent
  `:RELATES_TO {name: <type>, ...}` with `valid_at` ← `r.prov_valid_from`
  (or `r.created_at` if absent), other properties copied. Delete the
  original typed edge in the same query. No compat alias.
- **Single bitemporal naming convention** across canonical edges and
  claims: `valid_at` / `invalid_at` (event time) + `created_at` /
  `expired_at` (system time). Retire `prov_valid_from` / `prov_valid_to`.
- **Connector port shrinks to:** `kind`, `capabilities`, `list_artifacts`,
  `normalize_webhook`, `fetch`. Five verbs, read+webhook only.
- **Graphiti-keep sub-decision (revisitable).** Default: keep Graphiti for
  `add_episode` (audit log) and the Neo4j driver wrapper. Disable its LLM
  extraction (ignore output if no `extract=False` knob exists). **Revisit
  at P5** with cost telemetry: if `add_episode` is the only Graphiti
  function called and cost-per-episode is non-trivial, replace with a
  ~30-line custom episode-write function and delete the dependency. The
  POC proves nothing else in our stack depends on Graphiti.

### Implementation

1. **Delete dead deterministic webhook code.** Delete
   `SourceConnectorPort.propose_plan` and
   `SourceConnectorRegistry.propose_plan`. Delete the per-connector
   implementations (`connectors/github/plan.py`,
   `connectors/linear/plan.py`, the Notion stub) and their isolated unit
   tests. Delete the `context_graph: object` parameter on `propose_plan`'s
   signature; nothing reads it.

2. **Migrate to `:RELATES_TO` edges.**
   - Rewrite `canonical_writer.py`'s edge upsert path to emit
     `:RELATES_TO` per the POC's `write_claim` pattern (lines 157-205 in
     the POC). MERGE key: `(group_id, name, subject_key, object_key,
     source_ref)`. All bitemporal + provenance properties from day one.
   - Add a deterministic `render_fact(edge_upsert) → str` helper per
     predicate so `r.fact` is reproducible across re-writes. The agent
     supplies the entity natural-language names; the helper renders the
     canonical fact text.
   - Add the composite indexes per the POC: `RELATES_TO(group_id, name)`
     for filtering, `RELATES_TO(group_id, invalid_at)` for live-set
     traversal. Add an entity index on `Entity(group_id, entity_key)`.
   - One-shot Cypher migration script: walk every typed edge, write the
     `:RELATES_TO` equivalent, delete the original. Idempotent; safe to
     re-run.

3. **Re-point lifecycle machinery at the canonical graph.**
   - `apply_predicate_family_auto_supersede` is already a `:RELATES_TO`
     query — confirm it runs unchanged in the new world (POC follow-up
     test #1).
   - `apply_family_conflict_detection` likewise (POC follow-up test #2).
   - Move both to run **after canonical claim writes**, not after
     `add_episode`. They are now load-bearing, not shadow-graph
     bookkeeping.
   - Delete `classify_modified_edges.py` (was operating on Graphiti's
     shadow `:MODIFIED` extractions).
   - `ontology_classifier_pass.py` — assess whether it's still useful on
     the canonical graph. Likely simplified or deleted.

4. **Stop running Graphiti's LLM extraction consumers.** Disable
   extraction in `add_episode` if a knob exists; otherwise let it run and
   ignore the output. Either way: nothing in the read path or in
   lifecycle machinery looks at the resulting `:Entity` (UUID-keyed) or
   shadow `:RELATES_TO` nodes that Graphiti produces. Verify with a query
   that confirms shadow nodes accumulate but are never reached by any
   reader.

5. **Native Neo4j vector index setup.** Per POC T9:
   ```cypher
   CREATE VECTOR INDEX claim_fact_embeddings IF NOT EXISTS
   FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
   OPTIONS { indexConfig: {
       `vector.dimensions`: 1536,
       `vector.similarity_function`: 'cosine'
   }}
   ```
   Wire an embedding-on-write hook in `canonical_writer.py`: when an edge
   is written with a new or changed `fact`, enqueue an embed-and-set job.
   (Async; doesn't block the write.) `semantic_search` reader can move
   off Graphiti's `g.search()` onto `db.index.vector.queryRelationships`.

6. **Update `vision.md` and `architecture.md`** to reflect Position B
   and the narrowed Graphiti role.

### What does NOT change

- Deterministic `(group_id, entity_key)` entity identity. Position B
  preserves this unchanged.
- The four-tool agent contract.
- The hexagonal layout.
- The use of `add_episode` for episode persistence as the audit log.

### Cleanup

- Delete `docs/context-graph/plan.md` (this doc supersedes it).
- Delete `prov_valid_from` / `prov_valid_to` property writes everywhere.
- Delete the now-unused typed-edge label vocabulary in `canonical_writer.py`.
- Delete the `ontology_downgrades` carrier on `ReconciliationPlan` once
  E1 lands and downgrades become rare — track but defer to P3.

### Docs to update

- `vision.md`, `architecture.md` per above.
- `extending.md`: remove the `propose_plan` recipe; document the
  `:RELATES_TO` write pattern.
- This file: mark P0 complete.

### Exit criteria

- `grep -ri "propose_plan" app/src/context-engine docs/` returns nothing.
- `grep -ri "\\:OWNS\\|\\:DEPENDS_ON\\|\\:STORED_IN" app/src/context-engine/`
  returns only the typed-edge-to-`:RELATES_TO` migration script (which
  is then deleted after running) and the ontology's predicate-name
  constants. No typed-edge writes in `canonical_writer.py`.
- POC follow-up tests #1–#6 all pass against the live system.
- `prov_valid_from` / `prov_valid_to` do not appear in any new write
  path; existing data has been migrated.
- Vector index `claim_fact_embeddings` exists; at least one reader
  exercises it.
- One bench scenario asserts the post-migration substrate (e.g., a claim
  written via the new path, retrieved via the bitemporal predicate at
  `as_of`).

---

## Phase 1 — Identity contract (D2)

**Goal:** a Linear ticket, a PR title, a doc heading, and a k8s manifest
referring to "checkout v2" resolve to one `entity_key`, deterministically
where possible, through an inspectable alias table elsewhere. The agent
can ask "why are these the same?" and get an answer.

**Why now.** Every later phase loses its meaning if cross-source facts
don't converge onto shared entities. The claim refinements (P2) build on
identity; the source scanners (P4) need identity to merge their facts
with event-driven ones (P5); the readers (P9) need identity to answer
"everything about service X."

### Discovery

- For each entity kind in the (refined) ontology, classify identity:
  - **External-ID-anchored** (PR/Issue/Commit/Deployment/Comment):
    identity = source-system ID. Trivial.
  - **Slug + alias** (Service/Component/Feature/Person/Team/Repository):
    identity = canonical slug; cross-source convergence via alias table.
  - **Content-hash** (Decision / Note / Document body): identity = hash of
    canonical content for dedup.
- Audit current `entity_key` generation per type (`pydantic_deep_agent.py`
  ~116-120 instructs the LLM to mint keys). Catalog the patterns the LLM
  actually produces today.

### Decisions this phase

- **Identity classes** (above) and the canonical slug grammar per class
  (e.g. `service:<name>`, `feature:<slug>`, `repo:<provider>:<owner>/<name>`).
- **Identity authority — committed up front** (per the design-review
  critique):
  - First source that observes an external-ID-anchored entity wins.
  - For slug+alias entities, the canonicalization step picks the slug
    deterministically (slug normalization + first-observer wins); LLM-
    assisted matching for ambiguous cases goes through a confirm queue.
  - Content-hash entities: hash is identity, no authority needed.
- **Alias storage:** `Alias` itself is just another **claim** —
  `(:Entity)-[:RELATES_TO {name: 'ALIAS_OF', source_ref: ...,
  evidence_strength: ...}]->(:Entity {canonical})`. This rides on Position
  B's edge shape; no new representation. The alias is queryable with the
  same provenance/confidence story as any other claim.
- **Canonicalization step:** deterministic resolver chain — exact slug
  match → known-alias lookup → optional LLM-assisted resolution with a
  confirm queue. Runs inline in the agent's apply_plan.
- **Agent introspection:** a `context_search` extension (or dedicated MCP
  method) that returns the alias chain for an `entity_key` with
  provenance — traversing `:RELATES_TO {name: 'ALIAS_OF'}` edges.

### Implementation

1. Define `IdentityClass` enum + per-class slug-format validation. Move
   `entity_key` minting out of the LLM and into deterministic helpers
   keyed off the LLM-extracted *natural name* + scope. The LLM proposes a
   name; Potpie computes the key.
2. Canonicalization pass:
   - Stage 1, deterministic: exact match on slug + class.
   - Stage 2, alias claim lookup (`:RELATES_TO {name: 'ALIAS_OF'}`).
   - Stage 3 (gated, audited): LLM-assisted match with embedding
     similarity over names + context. Output goes through a review queue,
     not auto-applied. The reviewing agent or human confirms; on confirm,
     write an `ALIAS_OF` claim with `evidence_strength = "attested"`.
3. Backfill: scan the existing graph for plausibly-duplicate entities
   (same class, near-identical names). Generate a candidates list; do not
   auto-merge until the alias resolution has run with review.

### What does NOT change

- The canonical `(group_id, entity_key)` MERGE pattern. We're changing
  *how the key is chosen*, not *how it stores*.
- Existing entities keep their keys; aliases overlay.

### Cleanup

- Delete the empty `SYNONYMS` table and `normalize_entity_key`'s
  references to it. The alias-claim layer subsumes the function.

### Docs to update

- `architecture.md` — new "Identity" section.
- `extending.md` — how to add identity rules for a new entity class.
- `agent-contract.md` — document the alias introspection endpoint.

### Exit criteria

- A scripted test: ingest the same logical Feature from three sources
  (LLM-extracted from Linear ticket body, LLM-extracted from PR title,
  config-file-scanned from k8s annotation in P4); confirm exactly one
  `Feature` node with three `ALIAS_OF` claim edges.
- The agent can call a "show me the aliases for this entity" function
  and receive provenance.
- No new entity write skips the canonicalization step.

---

## Phase 2 — Claim model refinements (belief, decay, contradiction, coverage-gap)

**Goal:** the claim substrate already exists post-P0 (POC-validated). This
phase finishes the belief layer: per-source weighting, time-based decay,
equal-recency conflict surfacing, **edge-contradiction detection (F3)**,
**coverage-gap-aware confidence (F5)**, and the coarse confidence label
that the envelope returns.

**Why now.** P2 is the last load-bearing pillar before parallelizable
ingestion (P4–P6) and retrieval (P7–P9) can run. Without honest derived
confidence, proper conflict surfacing, and edge-contradiction detection,
every later phase reproduces the proper POC's failure modes: PREF leaks
superseded policies (F3), INFRA returns `high` confidence on 0% recall
(F5), and the agent learns to ignore the trust signals.

**What the substrate POC already settled (don't re-decide):**
- Claim representation = `:RELATES_TO` edge with the property set in the
  POC. Done.
- MERGE-key includes `source_ref` for corroboration semantics. Done.
- Supersession of newer-different-object claims = the Cypher in POC's
  `supersede_older_claims` (lines 215-225). Re-pointed to canonical graph
  in P0; runs unchanged.
- Point-in-time predicate = one Cypher clause. Done.
- Belief shape = aggregate live claims per `(subject, predicate)`, rank
  candidate objects by `max_strength + corroboration_bonus`. Done in the
  POC at 30 lines.

**What the proper POC newly requires (must build, F3 + F5):**
- **Edge-contradiction detection** that emits `SUPERSEDES` when source
  text doesn't explicitly say it. The substrate POC's time-based
  supersession only fires when a newer claim has a *different object*
  for the same `(subject, predicate)`. It does **not** infer "this new
  Policy supersedes that old one" from semantic content. That requires
  either an LLM judgement step (Graphiti's `dedupe_edges.resolve_edge`
  pattern) or deterministic ontology rules (singleton predicates auto-
  supersede by recency). The proper POC's PREF failure mode (F3 — leaked
  ADR-003) is the smoking gun.
- **Coverage-gap detection** in the belief layer so the envelope's
  confidence can't be `high` when underlying recall is low. The proper
  POC returned `confidence=high` on INFRA's 0% coverage scenario; the
  derivation only saw the hits it had, not what should have been there.

**What this phase still has to build:**

### Discovery

- Walk the POC's `beliefs_for` function. Identify gaps: no decay, no
  per-source weighting, no equal-recency conflict surfacing, no
  verification status as a confidence input.
- Inventory every site that computes confidence today. They all need to
  route through the new belief service.
- Sample real data to calibrate decay curves. A k8s claim from yesterday
  vs from six months ago — what should the decay coefficient look like?
  Pick a starter curve; tune from bench data later.

### Decisions this phase

- **Belief derivation lives in a service over Cypher**, not in Cypher
  itself. Simpler; profile later if it becomes a hot path.
- **Confidence formula (with coverage-gap cap, F5):**
  ```
  per_fact_score = (max_strength_per_object
            × decay(now - latest_observed_at, family_ttl_hours)
            × source_authority_weight(source_system))
          + corroboration_bonus(count of distinct sources)

  per_fact_label = "high"   if per_fact_score ≥ 4
                    "medium" if per_fact_score ≥ 2.5
                    "low"    if per_fact_score ≥ 1
                    else      "unknown"

  # F5: coverage gap cap
  envelope_confidence = min(
      max(per_fact_label across hits),
      coverage_cap(coverage.status, planned_families_returning_empty)
  )
  # coverage_cap returns "high" only when status=complete AND no
  # planned family came back empty. Otherwise capped at "medium",
  # and "low"/"unknown" when partial/empty.
  ```
  Per-source weights default to 1.0; high-trust sources (`k8s-scanner`,
  `codeowners-scanner`) ramp to 1.2; ambient sources (`slack-message`,
  `pr-body-llm`) cap at 0.7. Lock the starter weights; tune from bench.
- **Decay curve:** linear from full weight at `observed_at = now` to zero
  at `observed_at + family_ttl_hours × 2`. Corroboration extends the
  effective TTL by 50% per additional source.
- **Equal-recency conflict surfacing.** When two claims for the same
  `(subject, predicate)` have the same `valid_at` (within tolerance) and
  the same `evidence_strength`, neither supersedes; both stay live; the
  belief is `{"unknown", contributing_claims: [...]}` and a `QualityIssue
  {kind: "conflict"}` node is created (re-use `family_conflict_detection`
  output here).
- **Edge-contradiction detection (F3) — the new load-bearing capability.**
  Two paths, ship both:
  - **(deterministic) Ontology-rule supersession.** Refine the ontology
    in P3 to mark `singleton_predicates` (per-`(subject, predicate)`
    only one live object). New deterministic claim on a singleton
    predicate auto-supersedes prior live claims on same subject. Handles
    `OWNED_BY`, `DEPLOYED_TO`, `CURRENT_VERSION`, etc. No LLM call.
  - **(LLM-driven, port from Graphiti) `dedupe_edges.resolve_edge`
    pattern.** Per new claim, find candidate prior claims via
    semantic-similarity over fact text + endpoint match. Single LLM
    call asks: `duplicate?`, `contradicts?`. On `contradicts`, emit
    a `SUPERSEDES` claim and stamp the prior claim's `invalid_at`.
    Source: Graphiti's `edge_operations.py:622-824`. **Estimated port:
    ~200 LoC; the most valuable Graphiti capability to bring across.**
    Cost: 1 small-model LLM call per non-singleton edge write — track
    in cost telemetry.
- **Verification status as a claim attribute.** A separate `:RELATES_TO
  {name: 'VERIFIED', source_ref: <agent-or-human-id>}` claim attached to
  a fix raises that fix's confidence. Multiple verifications corroborate.
- **The agent envelope's `confidence` field** becomes
  `{high|medium|low|unknown}`, computed by the belief service. The
  numeric score is internal only. The coverage-gap cap (F5) prevents
  cosmetic `high` on sparse results.

### Implementation

1. `BeliefDeriver` service in `domain/` with `current(subject, predicate)`
   and `history(subject, predicate)` methods. The POC's 30-line function
   is the starting point; add decay, source weighting, coverage-gap cap.
2. Wire `BeliefDeriver` into the read path (P7 reads candidates from
   readers, P8 envelope builder calls `BeliefDeriver` to derive confidence
   for each dominant fact; envelope also calls a `coverage_cap` evaluator
   that downgrades top-level confidence when planned families came back
   empty — addresses F5).
3. **Edge-contradiction detection (F3) pipeline.**
   - Deterministic path: `singleton_predicates` registry in
     `domain/ontology.py`; canonical writer auto-stamps `invalid_at` on
     prior live claims for same-(subject, predicate) when writing a new
     deterministic-strength claim on a singleton predicate.
   - LLM path: implement a `port of dedupe_edges.resolve_edge` as
     `application/services/edge_contradiction.py`. Per new
     non-singleton edge: vector-search prior live claims with matching
     endpoints + similar facts; if any candidates, one structured-output
     LLM call returns `(duplicate_indices, contradiction_indices)`;
     emit `:RELATES_TO {name: 'SUPERSEDES'}` for contradictions, stamp
     `invalid_at` on superseded prior claims.
   - Both paths run in the same `apply_plan` step so the canonical
     writer is the only mutator.
4. Verification flow: when `context_record(record_type="verification")`
   lands (P6), the recording produces a `VERIFIED` claim on the target.
   `BeliefDeriver` consumes it in the corroboration count.
5. Telemetry: emit belief-score distributions and edge-dedup LLM-call
   counts per resolve so we can see when claims collapse to `unknown`,
   when contradiction-detection fires, and what it costs.

### What does NOT change

- The claim-as-edge representation (POC-settled in P0).
- The supersession mechanic (P0 re-pointed `temporal_supersede.py`).
- Identity (P1) is the substrate.

### Cleanup

- Delete `_confidence_for_coverage` and the three hardcoded constants
  (the cosmetic envelope-level confidence).
- Delete `CausalChainItem.confidence = 0.7` and any other hardcoded
  confidence defaults.
- Delete the soft-downgrade hardcoded `confidence: 0.3` once P3's
  ontology refinement makes downgrades rare; until then, treat it as a
  legitimate low-strength evidence flag.

### Docs to update

- New section in `architecture.md` — "Claims, beliefs, confidence."
- `agent-contract.md` — `confidence` is now coarse; explain the derivation.
- `bench-plan.md` — add scenarios that probe corroboration, supersession,
  decay (a stale fact must read as low-confidence without any explicit
  marker), and equal-recency conflict (must produce `unknown` belief +
  open conflict).

### Exit criteria

- Two ingestions of the same `(subject, predicate, value)` from different
  sources raise the derived confidence (extends substrate POC T3).
- Two ingestions with conflicting values at the same time produce an
  open `Conflict` (not auto-supersede).
- A claim observed beyond its family TTL with no corroboration reads as
  `low` or `unknown` confidence at query time (decay verified).
- A verification claim attached to a fix raises that fix's confidence
  one tier.
- **F3 fixed: a new policy claim that contradicts (by content, not time)
  a prior one emits `SUPERSEDES` and stamps the prior one `invalid_at`.**
  Bench scenario: ingest ADR-007 after ADR-003 (no explicit "supersedes"
  in source text); confirm the prior is auto-invalidated and PREF reader
  no longer surfaces it. Mirrors the proper POC's PREF failure case.
- **F5 fixed: a query that returns 0 hits for a planned evidence family
  produces `confidence ∈ {medium, low, unknown}`** in the envelope.
  Bench scenario: deliberately omit a fixture so INFRA returns sparse;
  envelope confidence caps at `medium` and `fallbacks` contains a
  `coverage_gap` entry.
- `confidence` in the envelope is never a hardcoded constant; trace any
  value back to a contributing claim's properties + coverage state.
- The benchmark's COMBO scenarios that probe trust signals (per
  `bench-plan.md`) flip from "always watch" to graded values.

---

## Phase 3 — Ontology refinement (E1)

**Goal:** ~30 entity types, all populatable. The Policy / BugPattern /
Fix shapes carry the structure their use cases need. Soft-downgrade
becomes rare because the schema matches what ingestion produces.

### Discovery

- Run the existing soft-downgrade telemetry against the last month's
  ingestion. Identify the labels and edges that downgrade most often;
  those are the highest-priority candidates for either deletion (nobody
  fills them at fidelity) or relaxation (required-properties too strict).
- For each of the 9 aspirational entities (`Initiative`, `Capability`,
  `RoadmapItem`, `Risk`, `OpenQuestion`, `Migration`, `Metric`,
  `Requirement`, `Feature` if it's not used): grep for any caller, any
  populator, any test that asserts on it. Confirm "no real use."

### Decisions this phase

- **Prune list** (confirm after discovery): the 9 above unless evidence
  surfaces a real populator. `Feature` is borderline — confirm whether
  any UC4-style "how does this feature work" query needs it; if so, keep
  but relax requirements.
- **Service ↔ Deployment join (F1) — named, required refinement.** The
  proper POC's INFRA 0% was driven by LLM extractors classifying k8s
  manifest entities as `Deployment` not `Service`. Add:
  - `Deployment-OF_SERVICE->Service` as a structured edge predicate (or
    `Deployment.service_key` as a deterministic property the canonical
    writer enforces).
  - `Service.deployments[]` reverse-lookup helper for the InfraTopology
    reader.
  - Identity-layer rule (in P1's vocabulary): when a Deployment entity
    is created with a name matching an existing Service slug, write an
    `OF_SERVICE` claim automatically.
  - The InfraTopology reader (P9) then hops `Service → Deployment →
    USES → DataStore` instead of expecting all topology edges to
    originate from the Service itself.
- **`singleton_predicates` registry** — for F3's deterministic-supersede
  path. Edges whose `(subject, predicate)` admits only one live object:
  `OWNED_BY`, `DEPLOYED_TO`, `OF_SERVICE`, `CURRENT_VERSION`,
  `POLICY_APPLIES_TO`-with-same-`code_scope`. Marked via a `singleton`
  flag on the edge spec in the declarative ontology.
- **`Policy` refinement:** add `policy_kind` enum (`error_handling`,
  `library`, `logging`, `naming`, `structure`, `testing`, `style`,
  `security`, `operational`, `process`), `code_scope` struct
  (`language[]`, `path_globs[]`, `role[]`, `framework[]`,
  `environment[]`), and link from `Policy` to the justifying `Decision`
  or `Incident`.
- **`BugPattern` refinement:** add `kind` enum (`dev_setup`, `build`,
  `ci`, `runtime`, `data`, `flaky`, `dependency`, `config`, `security`,
  `performance`), `symptom_signature` (canonical text; embedding is on
  the edges via P0's vector index), `scope_kind` (`global`, `language`,
  `stack`, `project`, `service`, `env`, `tenant`).
- **`Fix` refinement:** add `verification_status` enum (`reported`,
  `attempted_failed`, `worked_once`, `verified`, `recurred`),
  `recurrence_count`.
- **New claim predicates:** `ATTEMPTED_FIX_FAILED` (`Fix` → `BugPattern`,
  with reason) — first-class "we tried this, it didn't help."
- **Environment scoping — committed to edge property** (per the
  design-review critique). Topology claims (`name=DEPENDS_ON`,
  `STORED_IN`, `DEPLOYED_TO`, `USES`, `CONFIGURED_BY`) carry an
  `environment` property on the edge. Promote to a dedicated
  `EnvironmentBinding` only if the diff-between-envs query pattern
  proves it needs first-class subject treatment. Composite index includes
  `environment` for the dominant filtered traversals.
- **Required-property relaxation:** the only required property per entity
  is identity. Everything else optional; missing properties don't fail
  validation, they just produce lower-confidence claims.

### Implementation

1. Edit `domain/ontology.py` — prune, refine, add. The declarative
   machinery (`_build_*` derivations at the bottom of the file) handles
   the downstream rewiring automatically (this is the part of the design
   we keep).
2. Validation refactor: required-properties → identity-only; lifecycle
   enum validation stays but downgrade is for known-bad-value, not
   missing-value.
3. Migration: backfill `policy_kind` / `bug_pattern.kind` / `fix.
   verification_status` on existing data via a one-shot pass (LLM
   classification of existing free-text descriptions; review-gated for
   `must`-strength policies).

### What does NOT change

- The Activity-as-rollup model (UC3). PR/Commit/Deployment stay distinct.
- The Scope-as-interface design (Pot/Repo/Service/Env/Component/CodeAsset).
- The declarative-spec machinery.

### Cleanup

- Delete the 9 aspirational entities and any reader / classifier rule
  / test that referenced them.
- Delete `RELATED_TO` once soft-downgrade hits drop near zero (defer to
  bench data after a phase or two of operation).

### Docs to update

- `extending.md` — refresh the "adding an entity" recipe; document the
  prune / required-property philosophy.
- `agent-contract.md` — refresh the include catalog (this is partial; E4
  in P8 finishes it).
- `architecture.md` — the ontology section.

### Exit criteria

- Entity count ≈ 30.
- Soft-downgrade rate on a real ingestion corpus drops by >80% (bench
  measures this).
- Tests prove `policy_kind` / `bug_pattern.kind` / `fix.verification_status`
  drive the readers in P9.
- **F1 fixed: a k8s manifest scan produces both a Deployment entity AND
  an `OF_SERVICE` claim back to the named Service.** The InfraTopology
  reader's "what does auth-svc depend on in prod" returns the postgres-
  auth-prod DataStore. Mirrors the proper POC's INFRA 0% failure case.
- `singleton_predicates` registry populated; a new deterministic claim
  on a singleton predicate auto-supersedes prior live claims (works in
  concert with P2's edge-contradiction-detection).
- **Bench coverage average ≥ 55% across PREF/INFRA/TIME/BUG** (up from
  proper POC's 39.6% baseline; F1 alone should move INFRA from 0% to
  ≥ 50%).

---

## Phase 4 — Source-scanning ingestion (unblock UC2)

**Goal:** the infra topology entities have real data. K8s manifests,
dependency files, CODEOWNERS, GitHub Actions YAML feed
Service/Component/Environment/Deployment/Dependency/Person facts
deterministically with high evidence-strength.

> **Hard dependency on P5 (F2 fix):** the proper POC showed that scanner
> body-text alone isn't enough — CODEOWNERS extraction produced
> `(component:unknown)` because the LLM lacked file-path context. P5's
> path-aware scope-stamping (the deterministic activity layer) must be
> in place so that *the scanner stamps `scope.service`,
> `scope.environment`, `scope.repo` deterministically from the file
> path* before any LLM enrichment runs. P4 ships only after P5's
> scope-stamping is available. P4's LLM enrichment of body text adds
> nuance on top of the deterministic scope, not the other way around.

### Discovery

- Survey what config files exist in a realistic Potpie deployment (or a
  test pot): k8s/Helm directory structure, Terraform layout, presence of
  `pyproject.toml`/`package.json`/`Cargo.toml`, `CODEOWNERS`, `.github/
  workflows/`, OpenAPI specs. This determines the scanner priority list.
- Decide: scanner triggers are (a) per-commit on `main`, (b) scheduled
  tick, (c) on-demand. Lock for V1.

### Decisions this phase

- **New port:** `ConfigSourceScannerPort` — distinct from
  `SourceConnectorPort`. Verbs: `kind()`, `capabilities()`,
  `list_files(repo_ref) -> Iterable[ConfigFileRef]`, `parse_to_claims
  (file_content, file_ref) -> Iterable[Claim]`. No webhooks, no fetch
  (the scanner reads from a working tree).
- **Scheduling:** initial implementation = "on a fresh commit to main,
  enqueue a scan job per registered scanner per repo." A scheduled
  fallback for repos without webhook coverage.
- **Scanner registry:** in `application/services/`, parallel to
  `SourceConnectorRegistry`. Wire via `bootstrap/container.py`.
- **V1 scanner set:** `KubernetesManifestScanner`,
  `DependencyManifestScanner` (Python + JS + Go), `CodeownersScanner`,
  `OpenApiSpecScanner`. Defer Terraform / docker-compose / GitHub Actions
  to a follow-up unless the bench shows they're needed.

### Implementation

1. Port + registry per the decisions.
2. Each scanner returns Claims (the `:RELATES_TO` edge shape), not full
   plans. Claims have `evidence_strength="deterministic"`,
   `source_system=<scanner>`, `source_ref=<repo>:<path>:<commit_sha>`,
   `observed_at=<scan_time>`, `valid_at=<scan_time>` (unless the manifest
   carries a more specific effective-time annotation).
3. Claim writes go through the same canonical writer as agent-recorded
   ones. Re-scans update existing edges in place (MERGE-key idempotency
   from POC); when the topology changes, the new claim's
   `temporal_supersede` invalidates the old one automatically.
4. Identity (P1) ensures `Service:auth-svc` from a k8s manifest converges
   with `Service:auth-svc` from a PR description.
5. Job dispatching: reuse the existing batch queue or add a parallel
   "scan jobs" queue. Decide in discovery; recommend reuse.
6. Environment-scoping (P3): scanners stamp `environment` on edges
   derived from env-specific manifests (e.g., `k8s/prod/` →
   `environment="prod"`).

### What does NOT change

- The webhook ingestion path. Scanners are additive, scheduled, and
  orthogonal to event-driven ingestion.
- The LLM reconciliation agent. Scanners produce claims directly; the
  agent doesn't see scanner events.

### Cleanup

- (Once data flows) Low-confidence LLM-extracted topology claims that
  contradict scanner claims should be superseded automatically by the
  P0 lifecycle machinery. Verify; don't predict-delete.

### Docs to update

- New `extending.md` section: "Adding a config scanner."
- `architecture.md`: the four ingestion categories diagram.

### Exit criteria

- Test repo with a known k8s + dependency setup yields a populated
  `Service`+`Environment`+`DataStore`+`Dependency` subgraph that
  matches expectations.
- **CODEOWNERS scan produces `Service:auth-svc OWNED_BY Person:alice`
  (not `component:unknown`)** because P5's scope-stamping injected the
  Service context from the file path before LLM enrichment. F2 fixed.
- Re-scanning after a config change supersedes the prior facts (test
  validates bitemporal correctness via POC follow-up scenarios).
- Bench INFRA scenarios run against a real (non-mock) pot and pass at
  ≥ 70% coverage (vs proper POC's 0% — F1 + F2 together).

---

## Phase 5 — Event-driven deterministic activity layer + scope-stamping + MENTIONS (strengthen UC3, unblock P4)

**Goal:** every webhook event AND every source-scanner artifact gets a
deterministic scope stamp (verb, actor, subject, environment, file
path, occurred_at, source) **before any LLM call runs**. The LLM agent
only runs to enrich the *body* of free-form events — and even there,
into structured claims about the body, not entity upserts. Every entity
mentioned in an enriched body gets a `:RELATES_TO {name: 'MENTIONS'}`
claim back to the source episode for provenance traversal.

> **This is not undoing D0.** D0 deleted event-driven plan compilers
> (full ontology graph construction from a webhook payload). This phase
> introduces *deterministic scope stamping + activity claim creation*
> (small, narrowly-scoped, factual) as a separate step. The LLM still
> owns body-derived entity claims.
>
> **Promoted from "after P4" to "parallel with / prerequisite for P4"**
> based on proper POC findings F2 (CODEOWNERS extraction produced
> `component:unknown` without scope context) and F4 (PR-merged events
> didn't link to scope without MENTIONS-style provenance). P5 is now
> a prerequisite for P4 quality, not a downstream addition.

### Discovery

- For every webhook source we support (GitHub, Linear): catalog the
  structured-fact fields available in the payload (numbers, IDs, dates,
  actor logins, state transitions, labels). These are deterministic.
- Identify the cases where the LLM is currently doing trivial work an
  HTTP-payload parser would do better.

### Decisions this phase

- **Two-layer webhook normalization:** Layer 1 = `normalize_webhook`
  produces an `Activity` claim with deterministic scope (verb, actor,
  subject, environment-from-payload, occurred_at, source). No LLM.
  Layer 2 = the LLM agent runs only when there's body content
  (PR/issue text) and produces body-derived claims, **inheriting Layer
  1's scope as context** (the LLM is told "this event is about
  service:auth-svc in env=prod" rather than having to infer).
- **Path-aware scope-stamping for source scanners (F2 fix).** A scanner
  reading `clusters/prod/auth-svc.yaml` stamps every extracted claim
  with `environment="prod"` and `subject_hint="service:auth-svc"`
  deterministically from the file path. The LLM enrichment of the body
  inherits this context, so `(component:unknown) -[OWNED_BY]->
  (alice)` becomes `(service:auth-svc) -[OWNED_BY]-> (alice)` because
  the LLM was told the scope up-front.
- **MENTIONS provenance (F4 fix, Graphiti research finding).** Every
  entity surfaced in an enriched body gets a `:RELATES_TO {name:
  'MENTIONS', source_ref: <episode_uuid>}` claim back to the Episode.
  This lets TIME's reader query "what activities touched this scope"
  via `MATCH (a:Activity)-[:MENTIONS]->(target {entity_key: $key})`.
  Replaces or complements the `episode_uuid` property already on
  claims. The Graphiti porting research estimated ~50 LoC for the
  MENTIONS write path.
- **Activity claim shape:** Activity is an Entity (multi-labeled with the
  rich type when applicable, e.g., `:Entity:PullRequest:Activity`). The
  activity's outgoing claims are `:RELATES_TO` edges with
  `name="PERFORMED_BY"` to the actor, `name="TOUCHED"` to subjects,
  `name="MENTIONS"` to body-extracted entities. `verb_class` is an edge
  property (`code_change` / `deployment` / `discussion` / `decision` /
  `alert`).
- **Backfill via `list_artifacts`:** close the Linear stub. Each artifact
  enumerated yields an Activity claim with `observed_at=<backfill_time>`,
  `valid_at=<artifact_event_time>`.
- **Cost budget per webhook event:** Layer 1 (deterministic) costs ~$0.
  Layer 2 (LLM body enrichment) has a per-event LLM cost ceiling tracked
  via the telemetry from P0. If a single event exceeds the per-pot
  budget, Layer 2 is skipped and a `pending_enrichment` flag is recorded.

### Implementation

1. Refactor GitHub webhook adapter: structured parse → Activity claim
   first, *then* enqueue the body for LLM enrichment if non-trivial.
2. Same for Linear.
3. Implement Linear `list_artifacts` for issues + projects.
4. Add a deployment-event ingestion source. Options: GitHub Actions
   workflow-run webhooks (cheapest first), ArgoCD application status,
   manual `context_record(record_type="deployment_note")`. Pick one for
   V1.
5. Activity reader scaffolding (the full UC3 reader is in P9; here we
   just confirm activities are queryable).
6. **Graphiti-keep revisit.** If post-P0 telemetry shows `add_episode` is
   the only Graphiti function ever called and per-episode cost is
   non-trivial, this is the phase to swap it for a custom episode-write
   function and delete the Graphiti dependency. Decision is data-driven.

### What does NOT change

- The reconciliation agent (still LLM-extracts body content into claims).
- Connector contract (kind / capabilities / list_artifacts /
  normalize_webhook / fetch).

### Cleanup

- Remove any "LLM extracts the verb+actor+occurred_at from the PR
  payload" code; the deterministic layer owns that now.

### Docs to update

- `architecture.md`: the two-layer ingestion model.
- `extending.md`: how the activity layer interacts with `normalize_webhook`.

### Exit criteria

- A GitHub PR-merged event produces an Activity claim **without an LLM
  call** in the activity-layer path (verify via cost telemetry / span).
- **F4 fixed: a PR-merged event with body mentioning `auth-svc` produces
  a `:RELATES_TO {name: 'MENTIONS'}` claim from the PR Activity entity
  back to `service:auth-svc`.** TIME reader can find this PR by querying
  "activities mentioning auth-svc in last 7d". Mirrors the proper POC's
  TIME 25% failure case.
- **F2 fixed via path-aware scope-stamping: scanner-style scope
  inference works on a CODEOWNERS file at `apps/auth/CODEOWNERS`,
  producing `service:auth-svc OWNED_BY person:alice` deterministically
  before any LLM enrichment.** Unblocks P4.
- Linear backfill enumerates and ingests all open + recently-closed
  issues for a connected pot.
- Bench TIME scenarios that probe multi-source merge pass at ≥ 70%
  coverage (vs proper POC's 25%).

---

## Phase 6 — Structured recording UX (UC4 + UC1 primary path)

**Goal:** `context_record` accepts structured shapes per `record_type`,
so a recorded fix is queryable as a fix, a recorded preference is
filterable by `policy_kind` and `code_scope`. The agent records once and
the data is rich enough that retrieval works without LLM re-parsing.

### Discovery

- For each `record_type` in `CONTEXT_RECORD_TYPES`, catalog what the
  agent actually submits today (mostly free-text `summary` and `details`).
- Design the per-record_type structured payload. The HTTP body becomes a
  discriminated union by `record_type`.
- **Recording trigger** (per design-review critique): without an agent
  deciding to record, the bug graph stays empty. Pick an initial trigger
  mechanism. Options: a skill the agent invokes deliberately; a post-task
  hook the harness fires; mining git commits with `fix:` prefix for
  passive accumulation. Decide V1.

### Decisions this phase

- **Structured payload per `record_type`:**
  - `fix`: `{symptom_signature, root_cause?, fix_steps[],
     verification_status, kind, scope_kind, attempted_failed_fixes?[]}`.
  - `bug_pattern`: `{kind, symptom_signature, scope_kind, summary,
     reproduction_steps?}`.
  - `preference` / `policy`: `{policy_kind, code_scope, prescription,
     strength, justification_ref?, audience}`.
  - `decision`: `{title, summary, rationale, alternatives_rejected?,
     affects_refs?}`.
  - `verification`: `{target_ref, outcome ∈ {worked, didn't_work,
     partial}, notes?}` — attaches to an existing `Fix`.
  - … and the rest per their ontology slots.
- **Dedup on write:** when recording a `fix` or `bug_pattern`, run a
  symptom-similarity lookup against existing patterns first via the
  native vector index (POC T9). Either attach to existing or create new.
  Agent sees the decision in the response (e.g.
  `attached_to_existing_pattern: <key>`).
- **Recording trigger V1:** start with the explicit-skill path. The
  agent calls a recording tool when it has resolved something worth
  remembering. Add the harness-hook and the git-mining paths in a
  follow-up once we have data on what's worth recording.

### Implementation

1. Refactor `record_durable_context` use case + `ContextRecordRequest`
   model. The payload becomes a discriminated union.
2. Per-`record_type` claim emission: a recorded `fix` produces (a) a
   `Fix` entity with structured fields, (b) one or more `:RELATES_TO
   {name: 'RESOLVED'}` claim edges from `Fix` to `BugPattern`, (c)
   optional `:RELATES_TO {name: 'ATTEMPTED_FIX_FAILED'}` claims.
3. Dedup-on-write helper: symptom-similarity search via the P0 native
   vector index over `BugPattern` claim edges' `fact_embedding`.
4. Verification handling: a `verification` record produces a
   `:RELATES_TO {name: 'VERIFIED', source_ref: <recording-id>}` claim on
   the target `Fix`. The P2 belief layer consumes it in corroboration.

### What does NOT change

- The HTTP route `POST /record`. Body schema changes; tool surface
  doesn't.
- The four-tool agent contract.

### Cleanup

- Delete free-text-only record-type code paths.
- Delete the `context_ingest` 5th-tool advertising in the manifest (this
  is overdue; do it here).

### Docs to update

- `agent-contract.md` — full per-record_type payload schemas.
- `extending.md` — adding a new record_type.

### Exit criteria

- An agent records a fix with structured fields; the next agent querying
  the same symptom retrieves it with all fields intact and ranked
  appropriately.
- Two independent agents recording verifications for the same fix raise
  its derived confidence (test).
- Bench BUG scenarios that probe verification-aware retrieval pass.

---

## Phase 7 — Ranking layer (the missing pillar)

**Goal:** every retrieval scores candidate claims/entities by
`strength × recency × scope-overlap × corroboration × semantic-similarity`.
The same ranker serves every reader. Cosmetic "first 12 results" sort is
gone.

> The POC narrows this phase: **semantic-similarity** comes from native
> Neo4j relationship vector queries (T9), not a built-from-scratch hybrid
> search. The ranker is a thin weighting layer over inputs that mostly
> already exist on claim edge properties.

### Discovery

- Inventory the current "ranking" (mostly insertion order or per-leg
  caller-include order). Confirm there is no scoring today.
- Confirm the per-factor inputs are reachable:
  - `strength` — `r.evidence_strength` on each claim edge.
  - `recency` — `now() - r.valid_at` on each claim edge.
  - `corroboration` — count of distinct `(name, subject_key, object_key)`
    live claim edges.
  - `semantic-similarity` — score from
    `db.index.vector.queryRelationships` over `r.fact_embedding`.
  - `scope-overlap` — per-use-case computation (see per-factor note
    below).
  - **`coverage-quality`** (F5 follow-through) — whether the reader's
    planned scope returned hits at all. A hit-set assembled from a
    reader that returned `result.coverage = empty` is downweighted in
    the cross-leg merge so empty-leg results don't dominate when a
    paired leg succeeds. Feeds the envelope's coverage-gap cap from P2.

### Decisions this phase

- **Ranker contract:** `Ranker.rank(candidates: list[Candidate], context:
  TaskContext) -> list[RankedItem]`. `TaskContext` carries the agent's
  scope + intent + freshness preference.
- **Score formula:** weighted product (or weighted sum of logs) of the
  five factors. Lock starter weights; tune from bench data.
- **Per-factor signal extraction lives in each reader** (per design-
  review critique). The ranker is uniform; the upstream `Candidate`-
  building work is use-case specific:
  - UC1 scope-overlap: path-glob + language + framework intersection.
  - UC2 scope-overlap: entity-set membership.
  - UC4 scope-overlap: scope-hierarchy traversal (service > project >
    stack > language > global).
- **Reader contract update:** `ContextReader.read()` returns
  `Candidate`s (already including their per-factor inputs); the registry
  / orchestrator applies the ranker before merging into the envelope.

### Implementation

1. New `domain/ranking.py` with the formula and `RankingService`.
2. Refactor each reader to return scored candidates (each with the
   per-factor inputs the ranker needs).
3. The registry applies ranking and truncates by `max_items`.
4. Telemetry: emit per-resolve score distributions so we can see when the
   ranker is collapsing.

### What does NOT change

- Reader contracts (mostly — slight return-shape refinement).
- The envelope (per-include lists still come back, just ranked).

### Cleanup

- Delete per-reader ad-hoc sorting code; the ranker owns ordering.

### Docs to update

- `architecture.md`: ranking section.
- `bench-plan.md`: ranking-quality assertions per dimension.

### Exit criteria

- A bench scenario with two competing claims where the higher-strength /
  more-recent / better-scope-match wins, every time, deterministically.
- Cross-leg ranking: when the envelope mixes claims from different
  readers, ordering is by ranker score, not by reader order.

---

## Phase 8 — One read surface, task-anchored (E3 + E4)

**Goal:** one envelope shape across `goal=answer|retrieve` and across MCP
`context_resolve` / `context_search`. The agent vocabulary collapses to
~5 intents and ~10 task-shaped includes. Phantom includes either become
real or emit `unsupported_include`.

### Discovery

- Map every reachable envelope shape today (the three identified in PS7).
- For each phantom include (the ~30 silent no-ops): decide route — make
  real, fold into a sibling include, or emit `unsupported_include`.

### Decisions this phase

- **Intents (~5):** `feature` (UC1), `debugging` (UC3+UC4), `operations`
  (UC2), `onboarding` (UC2+dev-setup), `review`. Confirm during discovery.
- **Includes (~10):** `coding_preferences`, `infra_topology`,
  `recent_changes`, `prior_bugs`, `prior_fixes`, `decisions`, `owners`,
  `docs`, `source_status`, `semantic_search`. Tunable in discovery.
- **One envelope:** `bundle_to_agent_envelope` becomes the only shape;
  MCP `context_search` calls it through the same path; `goal=retrieve`
  goes through it (the reader registry produces the same bundle as the
  resolution service).
- **Mode reachability:** `deep` / `verify` reachable via MCP (extend
  `strategy → mode` mapping or expose `mode` directly).

### Implementation

1. Collapse the dual read abstractions (`IntelligenceProvider` + reader
   registry) into one path. Pick one as the trunk; rewrite the other to
   call it.
2. Rewrite MCP `context_search` to go through the trunk.
3. Update `CONTEXT_INCLUDE_VALUES` and `CONTEXT_INTENTS` per the
   decisions. Delete phantoms; route real ones via P9 readers.
4. Generate `agent-contract.md`'s include + intent catalog from code in
   CI. The doc never drifts again.
5. Cross-leg ranking via P7's ranker so the envelope's evidence is one
   coherent ranked list, not N independent buckets.

### What does NOT change

- The four-tool surface.
- The HTTP routes (paths stay; bodies/shapes consolidate).

### Cleanup

- Delete the two unused envelope shapes.
- Delete the `context_ingest` advertised tool (already done in P6;
  double-check no surface re-introduces it).

### Docs to update

- `agent-contract.md` — generated.
- `architecture.md` — read path becomes one diagram.

### Exit criteria

- `grep` shows one envelope builder, one read path.
- An unknown `include` value produces `unsupported_include`, never
  silent zero.
- `agent-contract.md` is generated; CI fails on drift.

---

## Phase 9 — Use-case readers

**Goal:** one reader per use case (and per future use case), built on the
primitives from P0–P8. Each reader contains **no logic specific to other
readers**; per-factor signal extraction is the only use-case-specific
work.

### Per-reader sketches

- **`CodingPreferencesReader` (UC1).** Inputs: task scope (language,
  path, framework, role). Logic: query `:RELATES_TO {name:
  'POLICY_APPLIES_TO'}` claim edges whose `code_scope` overlaps the task
  scope. Optional semantic-similarity hop via native vector index for
  fuzzy task descriptions. Rank via P7's `Ranker`. Returns: list of
  preferences with strength tier surfaced. Caches: per-(pot, scope-hash)
  for hot tasks.

- **`InfraTopologyReader` (UC2).** Inputs: scope (service / env / file).
  Logic: bounded neighbourhood traversal over claim edges with
  `name ∈ {'DEPENDS_ON', 'DEPLOYED_TO', 'STORED_IN', 'USES', 'EXPOSES',
  'CONFIGURED_BY'}`, environment-filtered via the `environment` edge
  property (P3). Supports blast-radius (incoming `DEPENDS_ON` traversal
  with depth). Supports `as_of` via the bitemporal predicate (POC T7).

- **`TimelineReader` (UC3, refactor).** Inputs: scope + window. Logic:
  multi-source Activity entities with their outgoing `PERFORMED_BY` /
  `TOUCHED` claim edges, scope-filtered, ranked by recency + scope-
  overlap + symptom-relevance (when query carries a symptom). Supports
  pulse rollups (`Period` aggregations).

- **`PriorBugsReader` (UC4).** Inputs: symptom signature (free-text or
  structured) + scope. Logic: **direct native vector query**
  (`db.index.vector.queryRelationships`) over `BugPattern` claim edges'
  `fact_embedding`, filtered to live edges and the agent's scope-or-
  broader. Per the POC T9 pattern. Rank by `verification × scope-
  overlap × recency × corroboration` via P7's `Ranker`. Surfaces both
  worked-fixes and attempted-failed-fixes (the latter labeled). Hides
  narrower-scope bugs from broader-scope queries.

### Implementation

- One reader per phase sub-step; each ships when its ingestion is enough
  to support it (P4 → InfraTopology, P5 → Timeline, P6 → PriorBugs and
  CodingPreferences).

### Exit criteria

- Each UC reader passes its bench scenarios.
- No reader contains logic specific to another reader (the LoC ceiling is
  removed; the constraint is structural separation).

---

## Phase 10 — Continuous benchmark validation (runs from P3 onward)

**Goal:** the system is judged against `bench-plan.md`'s
PREF / INFRA / TIME / BUG / COMBO taxonomy on both mock and API,
**continuously from P3 onward**. The mock score reflects retrieval
fidelity; the API score reflects ingestion fidelity. The gap is where
the work is. **The proper POC's 39.6% coverage average is the starting
baseline**; every later phase must hit its per-phase target (see the
proper POC findings section) before declaring done.

> **v4 change:** P10 is no longer a final-phase activity. The bench is
> the gating mechanism for every phase from P3 forward — phase exit
> criteria reference bench coverage explicitly. P10 is the activity of
> *running* and *maintaining* the bench, but the targets are checked
> per phase.

### Implementation

- Author the full scenario set per `bench-plan.md`. The proper POC's 4
  scenarios (one per dimension, with realistic distractors) and the
  substrate POC's 9 tests are the seed.
- Cover the corner cases that the design demands: corroboration ↑
  confidence; contradiction → SUPERSEDES (F3); stale → low-confidence;
  verification ↑ trust; scope-mismatch hidden; ranking deterministic;
  `as_of` past returns historical state; **coverage gap caps envelope
  confidence (F5)**; **Service ↔ Deployment traversal returns DataStore
  deps (F1)**; **CODEOWNERS scan produces Service-scoped ownership
  (F2)**; **PR-merged events findable by service via MENTIONS (F4)**.
- Per phase: at least one bench scenario asserting the phase's exit
  criterion is durably met. The F1–F5 fixes each need at least one
  bench scenario (these become the regression test set).
- Targets per dimension are committed in the proper POC findings
  section; first full run after P3 locks the absolute numbers per
  scenario; subsequent phases must improve them monotonically.

### Exit criteria

- All five dimensions ≥ targets on mock; the API/mock gap shrinks
  monotonically as P3 → P4 → P5 → P6 lands.
- F1–F5 regression scenarios are part of the CI bench corpus.
- A bench regression in CI blocks merges to `main` if any dimension
  drops > N% (N tuned in discovery; default 5%).

---

## Capability rollup by phase

| After… | Agent can demo end-to-end | Bench coverage target (avg) | Failure modes resolved |
|---|---|---:|---|
| P0 | Edge-shape migration complete: every canonical edge is `:RELATES_TO` with the bitemporal property convention. Native vector index live. Existing supersession + conflict machinery re-pointed at the canonical graph. Codebase smaller. | ≥ 40% (no regression) | — |
| P1 | Cross-source identity resolves; inspectable aliases (alias is itself a claim). Demonstrate "checkout v2" merging across sources. | — | — |
| P2 | Belief layer complete: corroboration, decay, equal-recency conflict surfacing, verification ↑ trust, **edge-contradiction detection (F3)**, **coverage-gap-capped confidence (F5)**. `confidence` becomes graded and honest. | — | **F3, F5** |
| P3 | Schema fits LLM output. Soft-downgrade rare. Policy/BugPattern/Fix carry structure. **Service ↔ Deployment join live (F1)**. Environment scoping live on topology edges. Singleton-predicate registry populated. | ≥ 55% | **F1** |
| P4 + P5 (parallel) | **UC2 works end-to-end** — infra topology populated from path-aware scanners; CODEOWNERS produces service-scoped ownership. **UC3 strengthens** — multi-source timeline merges PR + ticket + deploy + alert via MENTIONS provenance; backfill closes gaps; activity claims created without LLM. | ≥ 70% | **F2, F4** |
| P6 | **UC4 and UC1 work end-to-end via recording** — agents record structured fixes/preferences; retrieval surfaces them ranked, with verification status visible. | ≥ 75% | — |
| P7 | Every retrieval is ranked uniformly. Trust signals graded across all four UCs. Cross-leg merge respects coverage quality. | ≥ 80% | — |
| P8 | One envelope, one read path, one catalog (generated). MCP surface clean. `deep`/`verify` reachable. | — | — |
| P9 | Per-UC readers shipped. Each new use case ships as one reader file. | — | — |
| P10 | The system is benchmarked across all four dimensions continuously; the gap is honest and tracked. F1–F5 regression scenarios in CI. | bench-plan locked | — |

**F-codes refer to the proper POC failure modes** documented in the
"Proper POC findings" section. Each F maps to a specific phase
deliverable and a regression bench scenario in CI.

---

## Things explicitly deferred

- **Replacing Graphiti entirely.** The substrate POC made this look like
  a 2-week mechanical port. The proper POC + the Graphiti-internals
  research (see `pocs/position_b/findings_no_graphiti.md` and the
  Graphiti research artifact) revised the honest estimate **up to 3-4
  engineer-weeks**:
  - Substrate replacement (driver, episode write, vector index management):
    ~5-7 days (most of this is already in the POCs).
  - **Port `dedupe_edges.resolve_edge` + MENTIONS + temporal arithmetic**
    (~250 LoC from `graphiti_core/utils/maintenance/edge_operations.py:
    622-824`): 5-8 days. **This is load-bearing**: it's what emits
    `SUPERSEDES` when source text doesn't say it (F3). Skipping it
    regresses PREF and COMBO scenarios.
  - Bench-validation + extractor-prompt tuning to compensate for
    behaviors Graphiti was doing implicitly: 5-10 days.
  - Total: **3-4 weeks of focused work, bench-validated**, not "a P5
    sub-decision."
  - **Revisit trigger:** after P2's F3 implementation hits its bench
    target (≥ 70% PREF coverage). If our edge-contradiction-detection
    independently equals Graphiti's quality, removal pays for itself.
    If it doesn't, Graphiti is doing essential work and we keep the
    dependency.
- Cross-pot federation. One pot, one tenant.
- A frontend rebuild. The UI is a consumer of the agent contract.
- A "context_compaction" agent that prunes the claim store. Plausible
  P11; we wait until claim-store size becomes a real concern.
- The visualization tiers described in earlier discussions (Neo4j Browser
  cheat sheets are free with P0; embedded views per UC ship as their
  ingestion lands; cross-cutting views are a P11+ project).

---

## Working agreements for every phase

- Discovery findings → the phase's PR description, not the docs.
- Decisions made in discovery → `architecture.md` immediately.
- If a phase takes > 2 weeks of focused work, split it. If it's "almost
  done" for two weeks, it isn't — close the loop or replan.
- New tests for the new contract; old tests deleted with the old code.
- Every phase ships with at least one bench scenario that proves its
  exit criterion. No bench scenario = no phase exit.
- **Every phase from P3 onward must hit its bench-coverage target** (see
  the proper POC findings section + the sequencing table) before
  declaring done. A phase that doesn't move its target dimension up is
  the signal to stop, write down what changed, replan. This is the
  v4 gating mechanism.
- Doc + bench updates land in the same PR as the code. Never separate.
- **Cost is a first-class constraint.** Every ingestion phase budgets
  per-event/per-resolve LLM cost as part of its exit criteria. Telemetry
  from P0 makes this visible. P2's edge-contradiction-detection LLM call
  (the most expensive single addition) is tracked separately.

---

## v4 change log (what's new vs v3)

For traceability — what this revision folded in from the proper POC:

| Change | Source | Phase affected |
|---|---|---|
| F1 — Service ↔ Deployment join as required ontology refinement | Proper POC INFRA 0% | P3 |
| F2 — Path-aware scope-stamping promoted to P5 prerequisite for P4 | Proper POC CODEOWNERS extraction failure | P4 ↔ P5 dependency |
| F3 — Edge-contradiction detection as first-class P2 capability (port `dedupe_edges.resolve_edge` ~200 LoC) | Proper POC PREF leaked superseded ADR-003 | P2 |
| F4 — MENTIONS-style episode-to-entity provenance | Proper POC TIME 25%; Graphiti research finding | P5 |
| F5 — Coverage-gap cap on envelope confidence | Proper POC INFRA returning `high` on 0% recall | P2 + P7 |
| Bench coverage as continuous gating signal from P3 onward | All proper POC findings + bench-plan structure | All phases |
| Graphiti removal estimate revised 2 weeks → 3-4 weeks | Graphiti internals research + proper POC F3 finding | Things deferred |
| Per-phase bench coverage targets in sequencing table | Proper POC 39.6% baseline + bench-plan targets | Sequencing + capability rollup |

The v3 → v4 spirit: **substrate verdict survives intact; ingestion-
quality work is more substantial than v3 framed it.** The phases
themselves don't change identity — they get sharper requirements,
explicit cross-dependencies (P4 ↔ P5), and bench-coverage gating.
