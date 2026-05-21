# Position B proper POC — findings

> **Verdict.** The Position B substrate works mechanically (writes, reads,
> identity convergence, vector search, bitemporal queries). **The bottleneck
> is LLM-driven extraction, not the substrate** — coverage averages 39.6%
> across the four UC dimensions because the LLM doesn't reliably produce the
> entity-type, predicate, and subject-object directionality the readers
> expect. The previous POC hid this by hand-crafting claims; this one
> surfaces it by going through the actual ingestion path.
>
> Graphiti's contradiction-detection (which I dismissed as portable in 1 day)
> turns out to do real work that the bare extractor can't replace by prompt
> alone. The "delete Graphiti" recommendation needs to be qualified.

Run: `.venv/bin/python -m pocs.position_b_proper.main`

---

## Scorecard (4 scenarios, 1 per dimension)

| Dim   | Coverage | Precision | Phrase recall | Hallucinations | Confidence-match |
|-------|---------:|----------:|--------------:|---------------:|-----------------:|
| PREF  |    33.3% |     66.7% |         33.3% |        **yes** |             100% |
| INFRA |     0.0% |    100.0% |          0.0% |             no |             100% |
| TIME  |    25.0% |    100.0% |          0.0% |             no |             100% |
| BUG   |   100.0% |     66.7% |        100.0% |             no |             100% |
| **Avg** | **39.6%** | **83.3%** |  | | |

Honest reading: the substrate executed every query correctly. The *answers*
were partial because what got into the graph wasn't always what the reader
was looking for.

---

## What ran end-to-end

- 25 fixture events (3 universe seed + 15 signal + 7 distractors, mixed-source)
- LLM extraction (`gpt-5.4-mini`) over each event body → 79 claims emitted
- Identity resolution layer (slugify → alias lookup → embedding similarity →
  LLM-confirm → mint new) for every claim's subject + object → 72 entities
  created with an audited alias table
- Bitemporal `:RELATES_TO` writes per Position B
- Episode persistence with body embeddings for the BUG-style symptom query
- Four scenarios scored against expected `must_surface_event_ids`,
  `must_not_surface_event_ids`, key phrases, and confidence labels

Total runtime: 107s. ~$0.40 in OpenAI calls (estimate).

---

## What worked

1. **Substrate primitives.** Every `:RELATES_TO` write, alias write, episode
   write, vector-search query, semantic-search query, traversal — all ran
   correctly. No substrate bugs surfaced. `pocs/position_b/poc.py` and
   `poc_no_graphiti.py`'s 19 tests cover this; this POC re-exercises it
   under real ingestion.
2. **Identity convergence on natural-language variants.** The alias table
   correctly merged:
   - `auth-svc` + `auth service` → `service:auth-svc` (5 alias claims, 2
     distinct surfaces)
   - `inventory-svc` + `Inventory Service` → `service:inventory-svc`
   - `checkout-api` (4 consistent extractions)
3. **Bug-dimension retrieval.** UC4's killer query — semantic similarity
   over fact text + episode bodies — surfaced 100% of expected events
   (OPS-218 incident, PR #998 fix, postmortem, recurring-pattern Slack
   thread). This is the most validated UC; vector indexes on relationships
   and on episode bodies both pay off.
4. **Confidence label matched expectation in all 4 scenarios.** The coarse
   `{high/medium/low/unknown}` derivation from evidence strength +
   corroboration count produced the right label even when underlying
   coverage was poor. Caveat: that means the label is uncalibrated against
   *coverage* — see below.
5. **Distractor exclusion at the structural level worked.** INFRA's
   environment filter perfectly excluded staging claims when prod was
   requested (100% precision). TIME's window+scope filter perfectly excluded
   distractor PRs in unrelated services (100% precision).

---

## What broke — and what it tells us

### 1. INFRA: 0% coverage (the headline failure)

**Cause.** The LLM extractor classified `auth-svc` in k8s manifests as a
`Deployment` entity (key `deploy:auth-svc`), not a `Service`. So the claims
"DEPLOYED_TO prod" and "USES postgres-auth-prod" attach to
`deploy:auth-svc`, not to `service:auth-svc`. The INFRA reader queries
`MATCH (s:Entity {entity_key: 'service:auth-svc'})-[r]->()` — finds nothing
about postgres deps because those edges originate from the Deployment entity.

```
=== claims from k8s/auth/prod ===
  (deploy:auth-svc) -[DEPLOYED_TO]-> (env:prod)
  (deploy:auth-svc) -[USES]-> (datastore:auth-pg-prod)        ← needed
  (deploy:auth-svc) -[USES]-> (datastore:auth-redis-prod)     ← needed
=== claims from k8s/auth/staging ===
  (deploy:auth-svc) -[DEPLOYED_TO]-> (env:staging)
  (deploy:auth-svc) -[USES]-> (datastore:auth-pg-staging)
  (service:auth-svc) -[USES]-> (datastore:auth-pg-staging)    ← only one with service:
```

**What this is really telling us.** The LLM is doing a *reasonable*
classification — the k8s document literally describes a Deployment. The
ontology problem is that `Service` and `Deployment` are distinct types and
the relationships between them (`Service-HAS_DEPLOYMENT->Deployment`,
`Deployment-USES->DataStore`) need to be modelled and traversed. The
reader's "find dependencies of the service" must hop: `Service ↔
Deployment → USES → DataStore`. Mine doesn't.

**Fix categories:** (a) extractor prompt nudges to *always* lift to the
named Service when present (fragile); (b) ontology rule that
`Deployment.of_service` is a structured field so the reader can join
through it (correct); (c) a post-extraction normalization pass that
collapses `(deploy:auth-svc, USES, X)` → `(service:auth-svc, USES, X [env])`
when the deployment maps to a service (heavy but works).

**Estimated fix effort:** 2-3 days for option (b) — add `Service` ↔
`Deployment` mapping to the ontology, teach the reader to join.

### 2. CODEOWNERS extraction produced `(component:unknown)`

```
=== claims from codeowners/auth ===
  (component:unknown) -[OWNED_BY]-> (person:alice-acme)
    fact: * @alice-acme
```

The LLM saw `*` (the wildcard "everything in this directory") and
defensibly classified it as "component:unknown". But the contextual fact
— this is the auth CODEOWNERS — is lost.

**Cause.** Our extractor doesn't see *where the source file came from*, only
the body text. A k8s manifest at `clusters/prod/auth-svc.yaml` could tell
us "this is the auth-svc deployment in prod" deterministically, but the
extractor only sees the YAML contents. The deterministic-activity layer
from the rebuild plan (P5) is supposed to handle this: the *path* and
*structured metadata* of the source artifact are first-class, separately
from the LLM extraction of the body.

**Fix:** wire the deterministic activity layer. The CODEOWNERS scanner
should *know* it's scanning `apps/auth/CODEOWNERS` and stamp the subject
deterministically (`Service:auth-svc OWNED_BY Person:alice`), not ask the
LLM.

### 3. PREF: ADR-003 (superseded) leaked into preference retrieval

**Cause.** ADR-007 is supposed to *supersede* ADR-003. But the LLM extractor
emitted no `SUPERSEDES` edge from ADR-003's body ("SUPERSEDED by ADR-007"
in the text); it apparently only extracted the policy claim from ADR-003,
not the supersession fact. So my reader's `_superseded_subjects()` filter
returns empty, and ADR-003 surfaces.

**This is precisely the contradiction-detection capability Graphiti owns
that I dismissed.** Graphiti's edge resolution explicitly compares each
new claim against existing claims and emits invalidation edges. Without
that pipeline, supersession only happens when the *source text* explicitly
says "supersedes X" AND the extractor picks it up.

**Fix categories:**
- (a) prompt-tune the extractor to emit `SUPERSEDES` when source text
  contains "SUPERSEDED" / "deprecated by" / etc. Easy, brittle.
- (b) post-ingestion contradiction-detection pass: when a new claim about
  `(Policy:adr-007, GOVERNS, X)` enters, scan for existing
  `(Policy:adr-003, GOVERNS, X)` with overlapping scope and propose a
  `SUPERSEDES` link to the agent for confirmation. Like Graphiti's
  `dedupe_edges.resolve_edge` — needs LLM judgment.
- (c) deterministic ontology rule: any new Policy with `policy_kind=X` and
  overlapping scope supersedes prior Policies of same kind+scope. Brittle
  for unstructured cases.

I undersized this in the previous POC. **This is real work, and Graphiti's
implementation of it (~400 lines including the LLM dedup pipeline,
contradiction detection, and temporal arithmetic) is the most valuable
piece of the library for us to port — or to keep.**

### 4. TIME: 25% coverage — predicate/direction mismatch

The TIME reader queries claims where the service is the subject *or* object
with a small allowlist of predicates: `MERGED_BY, REVIEWED_BY, MODIFIED,
CLOSES_ISSUE, ADDRESSES, DEPLOYED_TO, AFFECTS, RESOLVED_BY`.

The LLM emitted, for the PR-merged event: `(person:alice) -[MERGED_BY]->
(pr:1042)`. PR #1042 isn't the service — it doesn't appear in the
"subject OR object = service:auth-svc" filter. The connection is the
*content* of the PR (rate-limiting auth-svc's login endpoint), not a
direct topology edge.

**Fix:** richer linking. Two paths:
- (a) Extractor emits a `(pr:1042, AFFECTS, service:auth-svc)` claim
  from the PR description. The LLM has the context to do this; it just
  needs explicit prompting.
- (b) Reader does a 2-hop query: "find activities whose extracted facts
  *mention* service:auth-svc" — needs the activity ↔ scope linking step.
  Either via a `MENTIONS` edge (which Graphiti writes — see the porting
  research) or via the `episode_uuid` property.

The Graphiti porting research flagged exactly this: "If you skip MENTIONS,
you lose provenance ('which episode said this?'). Port it." My substrate
stores `episode_uuid` on each claim, which gives episode → claim → entity
in one hop — equivalent to MENTIONS structurally. But neither my reader
nor my fixtures used it. The reader needs an "activities that touch this
scope via their episode body" query path.

### 5. Confidence label decoupled from coverage

All four scenarios returned `confidence=high` and matched the expected
label. *But* INFRA and TIME had 0–25% coverage. So a "high" confidence is
being reported on a response that's mostly missing the expected answer.

This is the same PS3 problem: the derived label reflects *what we returned*
(strength + count of hits) rather than *what we should have returned but
didn't*. A real engine needs to detect coverage gaps and surface them in
the envelope's `fallbacks` / `recommended_next_actions`. The current
derivation can't, because it doesn't know the recall ceiling.

---

## What this changes about the Graphiti decision

The previous POC's verdict was "Graphiti is dispensable, removal is cheap."
This POC sharpens that:

1. **The substrate replacement is still cheap** — the ~150 LoC of writes,
   reads, supersession, vector queries from `poc_no_graphiti.py` cover
   every load-bearing operation. The Graphiti porting research confirms
   ~1,300 LoC total for an equivalent system, dominated by extraction-side
   logic.

2. **Graphiti's contradiction-detection / edge-dedup pipeline is more
   load-bearing than I credited.** Look at sections 3 and 4 above —
   they're failure modes that Graphiti's edge resolution would have caught
   automatically:
   - Graphiti's `dedupe_edges.resolve_edge` LLM call asks "is this new
     fact a duplicate of an existing one, or does it contradict an
     existing one?" — that's how SUPERSEDES gets emitted today even when
     the source text doesn't explicitly say it.
   - Graphiti's `MENTIONS` edge writes every (episode, entity) link,
     which is the substrate the TIME reader's "find activities touching
     scope" query needs.

   **Without porting these two specifically, a Graphiti-removed system
   would be worse than Graphiti on the ingestion-quality axis.** The
   porting research estimated this piece at ~200 LoC (edge dedup) + ~50
   LoC (MENTIONS) — call it 1-2 engineer-weeks done properly.

3. **The Graphiti decision is now data-driven:**
   - **Keep Graphiti for ~1-2 phases.** Use its edge-dedup and MENTIONS
     while we build out source-scanners (P4) and improve the agent's
     structured emission (E1+P3).
   - **Re-evaluate after we have real coverage numbers.** If our LLM
     extractor + identity layer + (any port of edge-dedup) hits >70%
     coverage on the bench's 30 quick scenarios, removal pays for itself.
     If it doesn't, the LLM-dedup is doing essential work and we keep it.

4. **The previous "10-day estimate" was for substrate replacement only.**
   Honest re-estimate for *Graphiti removal that doesn't regress
   ingestion quality*:
   - Substrate replacement: 5-7 days (smaller than I said because I
     already wrote half of it in the POCs)
   - Port the edge-dedup + MENTIONS pipeline: 5-8 days
   - Validate against the bench corpus + tune extractor prompts to
     compensate for what Graphiti was doing implicitly: 5-10 days
   - **Realistic total: 3-4 weeks**, not 2.

---

## What the POC concretely validates and doesn't

### Validated

- Position B's `:RELATES_TO`-with-rich-properties edge shape supports
  every operation we asked it to: identity-keyed writes, bitemporal
  queries, semantic search over edge facts + episode bodies, env-filtered
  traversal, multi-source corroboration.
- The identity layer (slugify → alias → embedding → LLM-confirm) does
  converge cross-source name variants in practice. 2 of the 3 services
  had a natural-language variant correctly resolved.
- BUG-dimension retrieval (the killer UC4 query) works at 100% coverage
  with vector indexes alone.
- Honest scoring is possible: a bench-shaped scorecard with
  coverage/precision/hallucination/confidence-match is mechanically
  straightforward to author and produces actionable per-dimension
  signals.

### Not validated (and now sized honestly)

- **Coverage at production scale.** 39.6% average across 4 toy scenarios.
  Needs the full bench corpus (30 quick + 60 extended) to know whether
  the engine is at 40%, 60%, or 75% on real data.
- **Coverage under distractor noise at higher ratios.** Bench-plan says
  ratios go from 3:1 (easy) to 25:1 (hard). My POC was roughly 2:1.
- **Coverage on adversarial scenarios** (contradicting facts across
  sources). Not exercised here.
- **The 4 fixes for the 4 failure modes above.** Each is plausibly
  tractable but not built/measured.
- **Cost at production volume.** $0.40 for 25 events ≈ $0.016/event. A
  pot with 10k events ≈ $160 to ingest. Tractable but not free.
- **Graphiti removal doesn't degrade ingestion quality.** The whole point
  of the porting analysis is that some Graphiti capabilities ARE
  load-bearing. Removal requires re-implementing or replacing them.

---

## What I'd commit to next

1. **Run the actual bench scenarios** through this engine. The OPS-218
   BUG scenario is authored; 4 more easies are seeded. Get a real number
   per dimension before any further architecture decisions.
2. **Build the deterministic activity layer (rebuild-plan P5).** It would
   have fixed CODEOWNERS extraction immediately and would partially fix
   the k8s-manifest INFRA problem (the scanner knows the source path,
   environment, and target service from filename).
3. **Add Service ↔ Deployment ontology join** so INFRA reader hops
   through deployments. This is one of the ontology refinements P3
   already calls for.
4. **Port (don't build from scratch) Graphiti's edge-dedup loop** —
   it's ~200 LoC and is what would have emitted the missing SUPERSEDES.
   The porting analysis (`/Users/nandan/Desktop/Dev/graphiti`) names the
   exact files: `edge_operations.py:622-824` for the LLM-dedup call,
   `edge_operations.py:537-572` for the temporal arithmetic.
5. **Tighten the Graphiti-removal recommendation** to:
   *"Position B's edge shape is right; whether to remove Graphiti is a
   1-month staged decision, not a 2-week mechanical port. The substrate
   work is cheap; the equivalent-quality-extraction work is real."*

---

## Files

- `pocs/position_b_proper/fixtures.py` — Acme universe + 25 events + 4 scenarios with expected outcomes
- `pocs/position_b_proper/store.py` — bare-Neo4j write/read primitives
- `pocs/position_b_proper/identity.py` — alias resolution layer (the D2 implementation)
- `pocs/position_b_proper/extractor.py` — LLM-based claim extractor (structured outputs)
- `pocs/position_b_proper/readers.py` — per-dimension readers (PREF/INFRA/TIME/BUG)
- `pocs/position_b_proper/engine.py` — ingestion orchestration + envelope builder + scoring
- `pocs/position_b_proper/main.py` — runnable end-to-end
- `pocs/position_b_proper/findings.md` — this file
