# Context Engine Benchmarks — Plan & Tracking

> Living plan for what the bench measures, how it scores, and which instances exist.
> Pairs with `app/src/context-engine/benchmarks/README.md` (the how-to-run guide).
> Last reviewed for local-first docs: 2026-05-28.

This is a graph-quality benchmark plan, not the open-source packaging roadmap.
Use it to validate that local and managed graph backends return equivalent
agent envelopes for the same seeded corpus. The mechanism for that equivalence
is the adapter conformance suite: the seed/read scenarios run through the
`GraphBackend` interface (see
[`architecture.md`](./architecture.md#adapter-conformance)), so any backend —
in-memory, embedded SQLite, or Neo4j — is graded against the same contract.
Mentions of `--local` or `InProcessEngineClient` refer to benchmark execution
mode, not the final daemon-based OSS product shape.

The bench exists to give us a per-dimension score for the engine that survives
ontology / query-layer churn, so that the work of improving each layer shows up
as a score delta rather than a vibe. This doc defines the dimensions, the
scoring model, the synthetic-data strategy, and the harness changes required to
get there.

---

## 1. Goals and non-goals

### Goals

1. **Per-dimension scoring.** Each of the four use cases (below) tests a
   different *knowledge dimension* of the engine. Improvements to the
   reconciliation agent, the ontology, or a reader should land as a visible
   delta against the dimension they target — not be drowned in a single
   aggregate.
2. **Multi-source ingestion stress.** Every scenario draws events from ≥1
   source (GitHub, Linear, Slack, Notion, repo docs, alerting). The interesting
   ones draw from 3+ sources with deliberate overlap, conflicting facts, and
   out-of-order arrival, so the reconciliation agent is graded on extraction
   under noise, not on clean single-source feeds.
3. **Stable, long-lived corpus.** A small canonical synthetic universe (Acme
   Corp, §5.1) is reused across scenarios so that ontology improvements compound
   across the bench rather than each scenario optimising in isolation.
4. **Composite scenarios.** Real agent queries rarely sit in one dimension. A
   subset of scenarios deliberately spans 2–3 dimensions (e.g. "debugging an
   infra issue using timeline + bug-repo + infra knowledge") to catch the case
   where each dimension scores well alone but the engine cannot route across
   them.
5. **Extensibility.** Adding a use case is a new directory + a rubric template.
   Adding a source is a new connector module + a fixture folder. Adding an
   evaluator is a new file in `evaluators/`. No edits to the runner.

### Non-goals

- **Latency / throughput.** Separate concerns; not a quality benchmark.
- **LLM-quality benchmarking.** The judge is a measurement instrument, not the
  thing being graded. The engine is graded; the agent is a fixed harness.
- **Unit replacement.** Reader / connector unit tests stay where they are. The
  bench grades emergent behaviour, not contracts.
- **Production parity.** Synthetic universe is deliberately small and
  inspectable. Real-data benchmarks are a separate corpus (out of scope here).

---

## 2. Use case taxonomy

The four buckets are **knowledge dimensions** — what kind of knowledge the
engine has to hold and surface. They replace the previous task-shape taxonomy
(feature / debugging / review / operations / onboarding), which had 1 authored
scenario across 5 dirs and conflated "what is the agent doing" with "what is
the engine being tested on."

| Code | Use case | What it tests about the engine |
|---|---|---|
| `PREF` | Project preferences | Semantic recall of normative knowledge — error handling, file structure, library choice, logging style, naming, security rules — and the engine's ability to surface them *unprompted* on intent (e.g. "write a new endpoint" → log conventions appear). |
| `INFRA` | Project infra & architecture | Structural / topological knowledge — environments, deployment paths, service-to-service dependencies, secret stores, monitoring topology. Graph traversal correctness, not just lexical match. |
| `TIME` | Timeline | Temporal knowledge — what changed when, by whom, in what order; recent-change attribution; correlating a symptom to a window of churn. Tests ingestion-time ordering and recency-aware retrieval. |
| `BUG` | Bug / debug repo | Episodic / case-based memory — encountered failures, their root causes, and their fixes. Tests similarity matching ("have we seen this before"), case retrieval, and synthesis that proposes a fix grounded in prior cases. |
| `COMBO` | Composite (cross-dimension) | Engine routing across dimensions in one query — e.g. infra + timeline + bug. Not a fifth use case but a *modifier*: any scenario with `dimensions: [TIME, BUG, INFRA]` is graded per-dimension and the harness flags whether all required dimensions were exercised. |

### 2.1 Preferences (`PREF`)

**What good looks like.** Agent asks `intent=feature, scope={repo: web-app}`.
The engine surfaces:
- "Errors must be raised as `AcmeError` subclasses, never bare `Exception`."
- "Logging uses `structlog` with `event=` keys; no f-strings in log messages."
- "FastAPI handlers must declare response_model; no untyped dict returns."
- … without the agent having asked for any of these specifically.

**Sources.** README / CONTRIBUTING / `docs/conventions/*.md` ingested as
filesystem documents; PR review comments that crystallise into rules ("we now
require X"); Linear/Notion ADRs ("ADR-014: prefer pydantic over dataclasses for
DTOs"); Slack #eng-standards decisions.

**Why it's hard for the engine.** Preferences are normative and frequently
implicit — they accumulate through review comments and Slack threads, not as a
single doc. The engine must promote repeated guidance into stable rules and
de-duplicate near-paraphrases.

### 2.2 Project infra & architecture (`INFRA`)

**What good looks like.** Agent asks `intent=debugging, scope={service:
inventory-svc}`. The engine surfaces:
- Topology: `inventory-svc → postgres(inventory-prod-db) → kafka(orders-topic) → checkout-api`.
- Environment matrix: dev / staging / prod adapter choices, secret managers,
  monitoring backends.
- Deployment path: GitHub → ArgoCD → k8s (`inventory-prod` cluster).
- Owners: backend team `@platform-data`, on-call rotation.

**Sources.** Terraform / Helm / docker-compose files (filesystem); deploy
events (GitHub Actions, ArgoCD); README architecture sections; Notion
architecture pages; #infra Slack threads; CODEOWNERS / OWNERS files.

**Why it's hard for the engine.** The graph has to be both *complete* (no
missing edges between services that actually call each other) and *fresh*
(when a dependency is removed in a PR, the edge dies). Mostly tests graph
structure and reader correctness rather than synthesis.

### 2.3 Timeline (`TIME`)

**What good looks like.** Agent asks "what changed in checkout-api in the last
14 days that could touch the order-confirmation flow?" The engine surfaces:
- 3 merged PRs, ordered by merge time, with diff scope tags.
- 2 Linear issues that moved into Done in the same window, linked to those PRs.
- 1 deployment event (Wed prod release) and the post-release Sentry spike at
  +20 min.
- No mentions of changes outside the window or outside the relevant scope.

**Sources.** GitHub merge events; Linear state transitions; deploy events;
Slack incident channels (start / end markers); release tags.

**Why it's hard for the engine.** Ingestion-time correctness matters more than
synthesis — events must arrive with the right `occurred_at`, must order
correctly, and the engine must support recency-bounded retrieval. Distractors
(events outside the window or outside scope) must be excluded with high
precision.

### 2.4 Bug / debug repo (`BUG`)

**What good looks like.** Agent reports a fresh symptom. The engine:
- Finds the prior occurrence (same root-cause class, possibly different
  service), citing the identifier.
- Surfaces the fix that worked (PR / commit / runbook).
- Surfaces the *decision* (postmortem rule, if any) that should have prevented
  recurrence.
- Notes if today's setup violates that decision.

**Sources.** Linear bug / incident tickets; Sentry / DataDog alerts; postmortem
docs (Notion / repo `postmortems/`); Slack incident threads; PR descriptions of
fixes; runbooks.

**Why it's hard for the engine.** Bugs are *episodic* — each one is a coherent
story across many events. The engine must bind events into a case, then later
do similarity retrieval across cases. The OPS-218/220/389 scenario currently
under `use_cases/debugging/` is the canonical seed for this bucket.

### 2.5 Composite scenarios (`COMBO`)

A scenario declares `dimensions: [TIME, BUG, INFRA]` in its YAML. The harness:
- Verifies each declared dimension has at least one assertion (entities, edges,
  includes, or judge criteria tagged with that dimension).
- Scores the scenario once per dimension *and* once as an aggregate.
- Reports a composite scenario into each dimension's per-use-case rollup *and*
  into a separate `composite` panel.

Composite is where most production-shaped tests live. Roughly 25 % of the
scenario corpus.

---

## 3. Scoring model

### 3.1 Axes

Every scenario produces three **primary axes**:

| Axis | What it measures |
|---|---|
| **Ingestion** | After all fixture events have been reconciled, does the graph match the expected shape? Entities present, edges present, no soft downgrades, no failed events. |
| **Retrieval** | Did the engine surface the right facts in response to the query? Includes used, source refs cited, required event ids referenced, forbidden phrases absent. |
| **Synthesis** | Did the agent's answer (using only what the engine surfaced) satisfy the per-scenario rubric? Graded by the LLM judge against weighted criteria. |

Two **sub-axes** are graded alongside each primary axis and reported as
separate numbers in the report:

| Sub-axis | What it measures | Where it attaches |
|---|---|---|
| **Coverage** | Of the entities / facts that *should* have been recalled, what fraction were? | Ingestion (graph completeness) + Retrieval (recall of expected event_ids and includes). |
| **Precision** | Of the entities / facts that were surfaced, what fraction were relevant? Distractors injected per §5.3 are the negative class. | Ingestion (entities created that should not exist) + Retrieval (source_refs that match no expected event_id). |

Coverage and Precision are reported as percentages alongside their parent axis
score. A high primary-axis score with collapsing precision is a recall-without-
precision regression and surfaces as an amber flag in the diff view.

### 3.2 Axis weights per use case

Different use cases stress different axes. Weights for the aggregate score
per scenario are determined by `use_case` unless the scenario overrides:

| Use case | Ingestion | Retrieval | Synthesis | Rationale |
|---|---:|---:|---:|---|
| `PREF` | 20 | 40 | 40 | Ingestion is shallow (docs/comments → rules). Retrieval matters most — did the right preference surface? Synthesis matters — did the agent apply it? |
| `INFRA` | 30 | 40 | 30 | Graph structure must be right. Synthesis is descriptive rather than reasoned. |
| `TIME` | 40 | 30 | 30 | The whole point is "did the timeline come out correctly." Synthesis is usually enumeration. |
| `BUG` | 25 | 35 | 40 | Synthesis-heavy — case-matching is a reasoning skill, and "did the agent connect today's symptoms to OPS-218" is the headline metric. |
| `COMBO` | 30 | 35 | 35 | Balanced; the composite assertion logic (§2.5) is what matters. |

Per-scenario `axis_weights:` override is preserved (already in
`scenario.py:92`), but defaults now come from use case.

### 3.3 Pass criteria

- **Per axis**: score ≥ axis pass threshold (default 60 / 60 / 70).
- **Per scenario**: weighted aggregate ≥ scenario `pass_score` (default 70).
- **Per use case**: ≥ 80 % of scenarios in the bucket pass.
- **Per dimension on composite**: each declared dimension scores ≥ 60.

### 3.4 Aggregation & reporting

Reports already group `by_use_case` (`reporting.py:112-128`). Extensions:

1. Per-use-case panel now reports six numbers, not four:
   `count | aggregate | ingestion | retrieval | synthesis | coverage | precision`.
2. New `by_dimension` panel for composite scenarios.
3. New `by_source_mix` panel — see §5.2 for the mix tags. Lets us see whether
   single-source scenarios pass while multi-source ones fail (signals
   reconciliation-agent regressions).
4. New `by_difficulty` panel — see §4.3 — so a regression on `hard`
   doesn't get hidden by easy scenarios.

---

## 4. Scenario authoring

### 4.1 YAML schema (extensions to current)

Additions to the schema in `benchmarks/core/scenario.py`:

```yaml
id: <unique-snake-case-id>
use_case: PREF | INFRA | TIME | BUG          # new enum
dimensions: [PREF, INFRA]                     # optional; presence marks COMBO
tier: quick | extended
difficulty: easy | medium | hard | adversarial   # new
source_mix: single | dual | full | adversarial   # new; see §5.2

description: |
  One-paragraph statement of the agent task and why this is hard.

universe: acme                                # which canonical universe to seed from
seed:                                         # optional — pre-events for context
  - { event: universe/acme/services.yaml, at: "-365d" }
  - { event: universe/acme/team.yaml,     at: "-365d" }

ingest:                                       # scenario-specific events
  - { event: github/pr_merge__1042.json, at: "-14d", tags: [signal] }
  - { event: linear/issue_create__INV-103.json, at: "-14d", tags: [signal] }

distractor_events:                            # ingested but should not surface
  - { event: github/pr_merge__901.json, at: "-30d", count: 5, shape: "noise/random" }
  - { event: linear/issue_create__noise_*.json, at: "-21d..-7d", count: 12 }

post_ingest_assertions:
  graph_must_contain_entities: [...]          # unchanged
  graph_must_contain_edges: [...]             # unchanged
  graph_must_not_contain:                     # NEW — precision side
    - { label: Issue, key_pattern: "NOISE-.*" }
  reconciliation: { soft_downgrades_max: 0, failed_events_max: 0 }

query:
  intent: <intent>
  scope: {...}
  include: [...]
  mode: fast|balanced|verify|deep
  source_policy: references_only|summary|verify|snippets

retrieval_assertions:
  required_includes_used: [...]
  source_refs_min: 3
  must_cite_event_id: [linear/issue_create__OPS-218.json]   # now a list
  must_not_cite_event_id: [github/pr_merge__901.json]       # NEW — precision side
  forbid_in_answer: [...]
  temporal:                                                  # NEW — TIME-specific
    must_order_correctly: true
    window: { from: "-14d", to: "0d" }
    out_of_window_refs_max: 0

judge:
  pass_score: 70
  criteria:
    - name: <slug>
      weight: <int>
      pass_threshold: 1..5
      dimensions: [TIME]                       # NEW — which dimension this criterion grades
      prompt: "..."
```

The `dimensions:` field on a judge criterion lets composite scenarios attribute
the right judge score to the right dimension's rollup.

### 4.2 Rubric templates per use case

Each use case ships with a starter rubric in
`benchmarks/use_cases/<code>/_rubric.yaml`. Authors extend or replace it.
This keeps judge prompts consistent across the bucket and avoids each scenario
inventing its own scoring scheme.

Sketch of the four templates:

**`PREF/_rubric.yaml`**
- `cites_correct_preference` (30) — names the specific rule.
- `applies_preference_concretely` (30) — translates it into action.
- `no_invented_rules` (20) — every cited rule maps to a real source.
- `surfaces_unprompted` (20) — even when the user didn't ask, the right
  preference appears.

**`INFRA/_rubric.yaml`**
- `topology_correct` (35) — service deps stated correctly.
- `environment_distinction` (25) — prod/staging/dev not collapsed.
- `owner_attribution` (20) — names the right team / on-call.
- `no_invented_services` (20).

**`TIME/_rubric.yaml`**
- `correct_chronology` (30).
- `correct_window_bounds` (25) — nothing surfaced outside the window.
- `change_attribution` (25) — who, in what PR, when.
- `links_change_to_effect` (20).

**`BUG/_rubric.yaml`**
- `surfaces_prior_incident` (30).
- `identifies_recurrence_pattern` (20).
- `cites_decision_or_policy` (20).
- `actionable_first_steps` (15).
- `no_hallucination` (15).

The existing OPS-218 scenario already conforms to `BUG/_rubric.yaml`.

### 4.3 Difficulty ladders

Every use case has an instance at each difficulty level so we can plot a curve:

| Level | Characterisation |
|---|---|
| `easy` | Single source, signal-dense (low distractor ratio, ~3:1), recent. Engine has to do almost no reconciliation work. |
| `medium` | Two sources, moderate distractors (~10:1), historical context required. |
| `hard` | Three+ sources, heavy distractors (~25:1), spans months, requires linking entities across sources. |
| `adversarial` | Conflicting facts across sources (Linear comment contradicts PR description), out-of-order arrival, near-duplicate events. |

A regression on `hard` while `easy` stays stable is the most useful signal in
the bench — it means a reader or the reconciliation agent lost capability, not
a baseline shift.

---

## 5. Synthetic data plan

### 5.1 Canonical universe — "Acme Corp"

One stable fictitious organisation, seeded into every scenario so that
ontology improvements compound across the bench instead of each scenario
optimising in isolation.

**Org shape.**
- 5 backend services: `checkout-api`, `inventory-svc`, `payments-svc`,
  `notifications-svc`, `auth-svc`.
- 1 frontend: `web-app` (Next.js).
- 1 data service: `analytics-pipeline` (Airflow + Kafka consumer).
- 3 environments: `dev`, `staging`, `prod`.
- 1 monorepo (`acme/platform`) + 1 split repo (`acme/web-app`).

**People.** ~20 personas with stable handles and stable roles:
- Backend leads: `bench-user-1` (checkout), `bench-user-2` (inventory),
  `bench-user-3` (payments)
- SRE: `bench-user-4`, `bench-user-5`
- DBA: `bench-user-6`
- Frontend: `bench-user-7`, `bench-user-8`
- PM: `bench-user-9`
- EM: `bench-user-10`
- … (full table in §9)

Persona identifiers are *reserved* — the OPS-218 scenario already uses
`bench-user-1..5`. New scenarios pull from this pool, do not invent new ones,
so cross-scenario interference tests are meaningful.

**Stack.** Python (FastAPI, Pydantic, Celery), Postgres, Redis, Kafka,
ArgoCD, Helm, k8s, Datadog, Sentry. Picked because it matches Potpie's own
stack — fixtures stay easy to write and inspect.

**Timeline anchor.** "Now" in every scenario is *t=0*. Historical events use
relative offsets (`-90d`, `-365d`) so the universe slides forward naturally and
no fixture needs date-fixing as the project ages.

**Seed bundle.** `universe/acme/` holds the static seed:
- `services.yaml` — service catalog
- `team.yaml` — personas and roles
- `repos.yaml` — repo metadata + CODEOWNERS
- `architecture.md` — ADR-style architecture doc
- `runbooks/` — 5–10 runbook stubs
- `adr/` — 10 ADRs covering preferences and infra decisions

Seed events are ingested at `-365d` with low ingestion-quality bar (they're
context, not the thing being tested). The bench can run with `--seed-only` to
warm a pot for ad-hoc poking.

### 5.2 Source mix & connectors

Seven source types, each with a fixture directory and a connector module:

| Source | Connector | Event shapes | Used heavily by |
|---|---|---|---|
| GitHub | `github` (exists) | PR open/merge/close/comment/review, commit, release, issue | TIME, BUG, INFRA |
| Linear | `linear` (exists) | issue create / state change / comment | BUG, TIME |
| Slack | `slack` (new) | message, thread, channel-create | PREF, BUG, INFRA |
| Notion | `notion` (new) | page create / update, comment | PREF, INFRA |
| Repo docs | `repo_docs` (new) | filesystem doc create/update (ADR, runbook, README) | PREF, INFRA |
| Alerting | `alerting` (new) | Sentry / Datadog alert fire / resolve | BUG, TIME |
| Deploy | `deploy` (new) | ArgoCD / GHA deploy start / success / fail / rollback | INFRA, TIME, BUG |

`source_mix:` tag on each scenario controls how many of these are blended:
- `single` — one source. Used for the `easy` rung.
- `dual` — two sources, deliberately overlapping (a Linear issue + the PR
  that closes it). Used for `medium`.
- `full` — three or more, with cross-source linking required. Used for `hard`.
- `adversarial` — sources that contradict (issue comment says fixed, PR
  reverted; runbook says X, ADR says Y). Used for `adversarial`.

Reconciliation is the engine's product — the bench's job is to vary the source
mix until the agent has to work for it.

### 5.3 Distractor strategy

Every scenario specifies `distractor_events:` — events ingested but unrelated
to the query. Without distractors we'd grade only recall on small graphs;
precision would be invisible.

| Difficulty | Distractor ratio (noise : signal) | Distractor character |
|---|---:|---|
| `easy` | 3 : 1 | Random unrelated events in the same repo / time window. |
| `medium` | 10 : 1 | Mixed: random + some near-miss events (same service, different problem). |
| `hard` | 25 : 1 | Heavy near-miss: events that match on 2/3 attributes of the signal (same service + similar time, different problem). |
| `adversarial` | 25 : 1 + conflict | All of the above + at least one event that *contradicts* signal facts. |

Distractor fixtures live under `fixtures/raw_events/noise/` and are generated
programmatically (templated) — not hand-authored — so we can grow them cheaply.
Generator script: `benchmarks/tools/gen_distractors.py` (new, §6).

### 5.4 Instance counts per use case

Quick tier (CI; ≤ 90 s each) and Extended tier (nightly).

| Use case | Quick | Extended | Notes |
|---|---:|---:|---|
| `PREF` | 6 | 12 | Cover: error handling, logging, naming, framework choice, test conventions, security. One per domain at `easy`; ramp to `medium` / `hard` in extended. |
| `INFRA` | 5 | 10 | Cover: env layouts, service deps, deployment flow, secret routing, on-call topology. |
| `TIME` | 6 | 12 | Cover: recent-change attribution, multi-quarter drift, change-correlation-to-symptom, hot-spot detection, window edge cases. |
| `BUG` | 8 | 16 | Most variety: DB pool (OPS-218 seed), network, OOM, dependency conflict, auth, config drift, race condition, deploy regression. |
| `COMBO` | 5 | 10 | 2-dim and 3-dim composites; weighted toward the bug ↔ timeline ↔ infra triangle (the most common real query). |
| **Total** | **30** | **60** | |

30 quick is above the README's earlier "~20 quick" target — the extra 10 is the
cost of difficulty ladders (each use case wants at least one instance at each
of easy / medium / hard, plus one composite that touches it).

---

## 6. Harness changes needed

These are the code changes the plan implies. Listed for sequencing in §8, not
for immediate implementation.

1. **Taxonomy swap.** Edit `benchmarks/core/scenario.py:18` —
   `USE_CASES = {"PREF", "INFRA", "TIME", "BUG"}`. Add `DIMENSIONS` enum
   and validation that composite scenarios declare valid dimensions. Update
   `discover_scenarios` directory map.
2. **Schema extensions.** Add `dimensions`, `difficulty`, `source_mix`,
   `universe`, `seed`, `distractor_events`, `graph_must_not_contain`,
   `must_not_cite_event_id`, `temporal:` block, judge criterion `dimensions:`
   to the scenario loader.
3. **Sub-axis evaluators.** New `evaluators/precision.py` and
   `evaluators/coverage.py` that produce `(score, score)` tuples per primary
   axis. Wire into `core/result.py` so they appear in the report.
4. **Per-use-case axis weights.** Replace the global `AxisWeights` default with
   a lookup table keyed on `use_case`. Per-scenario override still wins.
5. **Distractor injection in replay.** `core/replay.py` accepts
   `distractor_events:` and interleaves them with signal events on the
   ingestion timeline, honouring `at:` ranges and `count:` expansions.
6. **Connectors for new sources.** Slack, Notion, repo_docs, alerting, deploy
   connectors under `adapters/outbound/connectors/`. Each connector parses
   the matching fixture shape and feeds the canonical event envelope.
7. **Universe seeder.** `core/universe.py` loads `universe/<name>/` into the
   pot at scenario start. Seed bundles are cacheable per pot fingerprint.
8. **Reporting extensions.** `reporting.py` grows the `by_dimension`,
   `by_source_mix`, `by_difficulty` panels. Markdown view adds a "deltas vs.
   baseline per use case per axis" table that already exists in spirit but
   does not include sub-axes today.
9. **Distractor generator.** `benchmarks/tools/gen_distractors.py` —
   templated event factories that emit N noise events matching a shape spec.
10. **CLI tweaks.** `python -m benchmarks run --use-case BUG --difficulty hard`
    and `--source-mix full` filters; `python -m benchmarks list --grid`
    renders the use_case × difficulty matrix so coverage gaps are visible.

These changes are additive — no existing assertion shape is removed, so the
single existing scenario (OPS-218, to be relocated to `use_cases/BUG/`)
continues to work without rewrite.

---

## 7. Reporting & dashboards

Three views, all rendered from the same JSON report:

### 7.1 Per-use-case panel (Markdown)

```
| Use case | N  | Aggregate | Ing | Ret | Syn | Cov | Prec | Pass |
|----------|---:|----------:|----:|----:|----:|----:|-----:|-----:|
| PREF     |  6 |      72.4 | 60  | 78  | 76  | 84  |  91  |  4/6 |
| INFRA    |  5 |      ...
```

### 7.2 Difficulty curve (Markdown)

```
| Use case | easy | medium | hard | adversarial |
|----------|-----:|-------:|-----:|------------:|
| PREF     |  92  |   78   |  61  |     —       |
| INFRA    |  ...
```

### 7.3 Source-mix curve (Markdown)

```
| Source mix | N | Aggregate | Δ vs baseline |
|------------|--:|----------:|--------------:|
| single     | 8 |      85.1 |         +1.2  |
| dual       | 9 |      71.4 |         −0.3  |
| full       | 8 |      54.7 |         −6.8  |
| adversarial| 5 |      32.0 |        −12.1  |
```

These three together tell a much more useful story than the current "aggregate
score" headline: which dimension regressed, on which difficulty, with which
source mix.

### 7.4 Baseline diff

Existing baseline-diff flag (`--baseline reports/main.json`) stays. It now
diffs every cell in every panel above, not just the headline.

---

## 8. Roadmap

Phased so each phase produces a usable improvement and the engine is
benchable end-to-end after Phase 2.

| Phase | Deliverable | Exit criterion | Status (2026-05-20) |
|---|---|---|---|
| **P0 — Taxonomy swap** | Replace USE_CASES enum, move OPS-218 to `use_cases/BUG/`, update reporting. | `python -m benchmarks list` shows 4 buckets; OPS-218 runs as a `BUG/hard/single-source` instance. | **Done** |
| **P1 — Canonical universe** | Author `universe/acme/` seed bundle + universe seeder. | Seeding a pot produces the topology, team, 3 ADRs, 2 runbooks without bench-time errors. | **Done** (8 seed envelopes; ramp to full 10-ADR bundle deferred) |
| **P2 — Schema + sub-axes** | Schema extensions, precision/coverage evaluators, distractor injection, per-use-case axis weights. | Scenarios score on all three primary axes + coverage + precision; report shows new panels. | **Done** |
| **P3 — Connectors** | Slack / Notion / repo_docs / alerting / deploy fixture connectors. | At least one fixture per new source, validated by `fixture validate`. | **Done** — fixtures land; engine-side stubs ship as `_bench_stubs.py` (passive plan-only) |
| **P4 — Quick corpus** | Author the 30 quick-tier instances (per §5.4). 1× easy + 1× medium per use case first; ramp. | All 30 scenarios load; ≥ 80 % execute end-to-end (pass rate unimportant at this stage). | **Done** (2026-05-25) — 30 quick-tier instances authored across PREF/INFRA/TIME/BUG/COMBO; all load + smoke clean. See the 2026-05-25 corpus expansion note below. |
| **P5 — Composite & adversarial** | 5 composite scenarios + adversarial ladder rung for each use case. | Composite scenarios score per declared dimension; adversarial scenarios run without crashing the reconciliation agent. | **Done** (2026-05-25) — 7 COMBO (per-dimension) + adversarial rungs for BUG/INFRA/TIME. See the 2026-05-25 corpus expansion note below. |
| **P6 — Extended ramp** | Ramp to 60 instances; nightly CI job. | Nightly run produces a report; baseline diff hooked up. | **In progress** — 48 of 60 (18 extended landed 2026-05-25); baseline-diff renderer + smoke gate already in; nightly CI still planned |

### 2026-05-20 operational pass (Tier 1/2/3 follow-up)

Triggered by review findings from running the bench end-to-end against the live engine:

- **`probe` subcommand.** New `python -m benchmarks probe` runs in <15 s: hits `/context/status`, confirms expected connector kinds, submits one minimal envelope per kind and waits for it to leave `queued`. Compresses the 10-minute "is the queue dead?" finding into a single command.
- **`smoke` subcommand.** New `python -m benchmarks smoke` runs every scenario through the bench-side pipeline (fixture resolution, timeline assembly, evaluator math, reporting) without an engine. Sub-second; right gate for every PR.
- **Engine lifecycle inlined into ingest errors.** Failed events now carry `lifecycle_status / stage / job_id / step_error / reconciliation_runs` in their error message; the ingestion evaluator surfaces those directly in the report so "is it the agent, the queue, or my fixture" is answered without a second debugging round.
- **Within-scenario parallel ingestion.** Seeds + distractors run in a 6-worker thread pool (signals stay sequential because they encode arrival order). Universe warmup goes from ~30 s × N to ~30 s flat per scenario.
- **Cross-scenario concurrency.** `python -m benchmarks run --concurrency N` fans scenarios across N workers (pots are isolated). 30-scenario quick tier becomes minutes, not half an hour.
- **Difficulty-adjusted default `pass_score`.** Scenarios without an explicit `judge.pass_score` pick from `easy:75 / medium:65 / hard:55 / adversarial:45`. The same answer no longer satisfies a different bar on different difficulty rungs.
- **Structured `must_cite_event_id` matching.** The retrieval evaluator matches `source_refs[].source_id` against the fixture envelope's canonical `source_id` instead of substring-matching the answer haystack. Substring search remains as a fallback only when the index is empty.
- **`coverage_floor` / `precision_floor` gates.** Scenarios can opt into sub-axis gating; ingestion or retrieval axis fails if coverage/precision drops below the declared floor regardless of the primary score.
- **`includes_actually_used` via `CoverageReport.available`.** The bench now consults the engine-declared coverage report (which already existed in `IntelligenceBundle.coverage.available`) instead of static key-name pattern matching. Static fallback kept for older response shapes.
- **Baseline-diff renderer.** `python -m benchmarks report current.json --baseline prior.json` now produces a markdown diff per panel cell, flags regressions on > −2.0 delta, and exits non-zero on any regression. The "ontology change → score delta" loop is wired end-to-end.
- **`new-scenario` scaffolder.** `python -m benchmarks.tools.new_scenario --use-case PREF --difficulty medium --id pref_xxx` writes a schema-valid stub from a per-use-case rubric template. Authoring cost drops from 50–150 lines of boilerplate to 1 command + content editing.
- **Stub connector enrichment.** `RepoDocsStubConnector` now does content-aware extraction: 20 `Person` entities from CODEOWNERS, 7 `Service` entities from the architecture doc, `Decision` per ADR. PREF/INFRA scenarios now have structural facts to grade against (previously every assertion failed because only `Document` entities reached the graph).
- **Per-dimension hooks on entity/edge assertions.** Schema now accepts `dimensions: [...]` on `EntityAssertion` / `EdgeAssertion`. Forward-compat for the future per-dim ingestion roll-up; runner still broadcasts today.
- **Unit tests for all new surfaces.** 39 / 39 passing across the bench test directory (`test_difficulty_thresholds`, `test_must_cite_structured`, `test_subaxis_gating`, `test_baseline_diff`, `test_smoke_pipeline`, `test_stub_enrichment` + the pre-existing four).

**What's still deferred:**
- `--reuse-pot` flag for cross-scenario universe-seed dedupe (parallel-within-scenario already captures the 80 % win).
- Per-dimension *runtime* roll-up of ingestion/retrieval scores (schema is in place; the runner change isn't).
- `--ablate reconciliation` ablation mode (needs engine-side header support).
- Real-data corpus anchor (P6 nightly + capture-and-redact tooling).

### 2026-05-26 schema-independent pass

Triggered by the recognition that the per-rubric judge and the deterministic ingestion / retrieval evaluators were measuring the **engine's data model** (graph labels, include vocabulary, fixture ids) almost as much as the agent's actual answer quality — so every ontology / read-trunk refactor moved the score. The pass adds an **invariant** grading mode and a **light** runner subset:

- **`evaluators/llm_judge_invariant.py`.** New schema-independent judge. Inputs: the scenario's `signal:`-tagged envelopes (full JSON) + counts of seed/distractor events (without content) + the agent's full answer. One LLM call returns four 0..100 sub-scores (faithfulness / coverage / clarity / usefulness), weighted 30/30/20/20 → aggregate. The engine's `coverage`/`includes`/`items` shape is *not* shown to the judge — the schema-independence is enforced at prompt-construction time.
- **Runner branch in `runner.py`.** `run_scenario(..., invariant=True)` keeps the full data path (ingest + reconcile + query) but: (a) ingestion + retrieval axes drop their gating role (they're recorded in the report as diagnostics only), (b) synthesis is replaced by the invariant judge, (c) aggregate = invariant score, (d) `_build_by_dimension` broadcasts the invariant score across declared COMBO dimensions.
- **CLI: `--invariant` flag** on `run` (any filter combo) + new top-level **`run-light`** subcommand that selects scenarios tagged `light: true`, defaults `--concurrency 5`, and turns invariant on by default. Five-way parallel, schema-independent, end-to-end.
- **Light subset (5 scenarios).** `pref_logging_structlog` (PREF/easy), `infra_topology_basic` (INFRA/easy), `time_recent_changes_window_14d` (TIME/easy), `bug_redis_connection_flap` (BUG/easy), `combo_onboarding_new_engineer` (COMBO/medium). New `light: bool = False` field on `Scenario`; loader picks it up; `list --json` surfaces it. A corpus test asserts exactly one light scenario per dimension so the curated subset can't drift.
- **Tests: 16 new** across two files. `test_invariant_judge.py` exercises the prompt builder (asserts engine-internal field values like `ready` / `prior_bugs` don't leak into the judge prompt), the JSON parser (strict / fenced / clamped / unparseable), the weighted aggregate, the empty-response short-circuit, and the unparseable-response failure path with a stubbed OpenAI client. `test_light_subset.py` exercises the loader field + the per-dimension corpus invariant. **55 / 55 bench tests pass.**

This is the first signal in the bench that survives schema churn by construction: changing an include name or splitting a node label doesn't move the invariant score because the judge never sees those concepts.

P0 + P1 + P2 is the smallest valuable beachhead — those are done, so
every ontology change to the engine now produces a clean score signal.
P3 onward is corpus expansion that runs in parallel.

### 2026-05-25 corpus expansion (P4 + P5 + extended ramp)

The corpus went from **5 authored scenarios → 48** (30 quick + 18 extended),
clearing the §5.4 quick-tier target (30) and starting the extended ramp.
All 48 pass `python -m benchmarks smoke` (fixture resolution + timeline
assembly + evaluator math + reporting) and `python -m benchmarks fixture
validate`. Coverage by use case × difficulty:

| Use case | easy | medium | hard | adversarial | total |
|---|---:|---:|---:|---:|---:|
| PREF  | 2 | 5 | 3 | 0 | 10 |
| INFRA | 1 | 4 | 3 | 1 | 9 |
| TIME  | 2 | 4 | 2 | 1 | 9 |
| BUG   | 4 | 4 | 4 | 1 | 13 |
| COMBO | 0 | 3 | 4 | 0 | 7 |

- **Universe ramp.** Added ADR-002 (testing), ADR-031 (security/input
  validation), ADR-045 (DB migrations + the +25% capacity rule, formalising
  the OPS-220 lesson), and an environments-and-secrets matrix doc to
  `fixtures/raw_events/universe/acme/` (8 → 12 seed envelopes). Added four
  clone-mode distractor templates under `fixtures/raw_events/noise/`
  (github / linear / slack / alerting).
- **Connectors exercised for real.** The previously-empty `notion/`,
  `alerting/`, and `deploy/` fixture dirs are now populated; every scenario
  draws from the source mix its difficulty rung demands (single → full →
  adversarial). ~110 fixture envelopes total.
- **BUG (13).** Authored: pool-exhaustion v2 (multi-source OPS-218→OPS-389),
  OOM-consumer, redis-flap, pydantic v1/v2 conflict, JWT-expiry race,
  staging↔prod config drift, **celery worker-starvation (adversarial: two
  near-miss priors, only one correct)**, plus extended n+1, CORS, Stripe
  webhook idempotency, disk-full-from-debug-logging, kafka rebalance storm.
- **PREF (10).** error-handling/AcmeError, pydantic-over-dataclass, test
  layout, **naming/style accumulated from review comments (no ADR)**,
  security input validation, plus extended api-versioning, db-migration,
  async-celery, dependency-pinning.
- **INFRA (9).** env-adapter matrix, deploy flow, secret routing, on-call
  topology, plus extended kafka topology, cache strategy, data-flow lineage,
  and an **adversarial service-dependency-removal (fresh vs stale doc)**.
- **TIME (9).** PR↔issue attribution, deploy→symptom correlation, multi-
  quarter drift, hotspot detection, **window edge-cases (±1h of the 14d
  boundary)**, plus extended release cadence, incident-timeline reconstruct,
  dependency-bump history.
- **COMBO (7).** debug-recent-infra-change (TIME+BUG+INFRA), postmortem
  drafting (BUG+TIME), release readiness (TIME+INFRA), decision-lookup-with-
  recency (PREF+TIME), incident-to-prevention (BUG+PREF), capacity planning
  (INFRA+TIME) — each per-dimension graded.

**Live end-to-end status (2026-05-25, branch `feat/ce-observability`).**
The bench runs against a live engine end-to-end (pot create →
`/events/reconcile` → graph snapshot → `context_resolve` → report → drop);
reconciliation runs for real (the `pydantic-deep` agent on `gpt-5.4-mini`,
real entity/edge mutations). After the read-trunk consolidation (P8/P9) three
alignment fixes were needed and landed (2026-05-25):

1. **Status endpoint (engine).** `report_status.py` referenced the removed
   `ContextEngineContainer.resolution_service` and `.readers` attributes →
   `/context/status` 500'd (blocked `probe`). Repointed to `container.context_graph`
   and derived the reader manifest from `READER_BACKED_INCLUDES`. `probe` is
   now **READY** (all 7 connectors registered, all drain checks pass).
2. **Graph snapshot (bench).** The read trunk answers every read as an
   `AgentEnvelope` (`result.items[]`), not `result.nodes/edges` — so the
   old snapshot parser saw 0 entities. `core/graph_inspect.py` now
   reconstructs entities + edges from the envelope's claim items
   (`payload.subject_key/object_key/predicate`; label from key prefix),
   querying **unscoped** with the reader-backed includes for the full topology.
3. **Retrieval evaluator + include vocab (bench).** The engine's read vocab is
   now `{coding_preferences, infra_topology, timeline, prior_bugs}` + planned
   `{owners, decisions, docs}`; scenarios used the old vocab (`service_map`,
   `prior_fixes`, …) so nothing routed to a reader. `evaluators/retrieval.py`
   now reads the envelope shape (`coverage`/`items`/`answer`), and all 48
   scenarios' `query.include` / `required_includes_used` were migrated to the
   reader-backed vocab.

Verified live: `infra_topology_basic` scores **ingestion 100 / retrieval 100**
(`includes_used=['infra_topology']`, cites the architecture doc, 0 failed
events, 41 entities / 50 edges).

**In-process harness (benchmark only; recommended; non-blocking, no :8001).** A new driver
(`benchmarks/core/local_engine.py`, `InProcessEngineClient`) runs the bench
**without the HTTP server on :8001 and without a Celery worker** — it builds
the engine container in-process and reconciles inline against the shared
Postgres/Neo4j via the same `handle_process_batch` verb the worker uses.
Activate with `python -m benchmarks run --local` (or `POTPIE_BENCH_INPROCESS=1`).
This frees the app's port and is self-contained for the **current benchmark
stack** (Postgres + Neo4j + `OPENAI_API_KEY`). It is not the target OSS local
daemon shape, which should work without a mandatory Postgres/Neo4j/Docker stack
or daemon-side model key. Being single-threaded, it also sidesteps the "Event
loop is closed" race below entirely. Verified: `--local` on
`infra_topology_basic` reconciles with 0 failed events and retrieval 100.

**Operational notes for the HTTP path:**
- Run the Celery worker with `--pool=solo --concurrency=1`. Under the
  `threads` pool, concurrent reconciliation batches hit an
  *"Event loop is closed"* asyncio-across-threads race in the `pydantic-deep`
  agent (a separate engine concurrency bug) that fails most events; `solo`
  serializes and reconciles cleanly. (The in-process harness avoids this.)
- Use `--ingest-timeout` ≥ 600 — the 12-seed universe makes per-scenario
  warmup heavy under per-event LLM reconciliation.
- Synthesis (judge) axis needs an engine answer synthesizer
  (`CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL`); without it `goal=answer` returns
  a deterministic fallback summary and the judge grades thin.

### 2026-05-20 implementation pass

- `scenario.py` rewrite: new enums (`USE_CASES`, `DIMENSIONS`,
  `DIFFICULTIES`, `SOURCE_MIXES`), schema fields (`dimensions`,
  `difficulty`, `source_mix`, `universe`, `seed`, `distractor_events`,
  `graph_must_not_contain`, list-form `must_cite_event_id`,
  `must_not_cite_event_id`, `temporal:` block, judge criterion
  `dimensions:`), per-use-case axis-weight defaults.
- `result.py`: `AxisScore` widened with `coverage` / `precision` fields;
  new `DimensionScore`; `BenchmarkReport` gains `by_dimension` /
  `by_difficulty` / `by_source_mix` panels.
- `evaluators/coverage.py`, `evaluators/precision.py`: sub-axis math
  consumed by the primary evaluators.
- `evaluators/ingestion_quality.py`: now computes coverage + precision
  and honours `graph_must_not_contain`.
- `evaluators/retrieval.py`: list-form citations, distractor citations,
  TIME window check, coverage/precision.
- `evaluators/llm_judge.py`: per-criterion dimension attribution +
  `synthesis_by_dimension()` helper.
- `core/replay.py`: distractor expansion (enumeration + clone modes),
  range offsets (`-21d..-7d`), `assemble_timeline()` orders seeds /
  signals / distractors by `(time, role)`.
- `core/universe.py`: discovers `fixtures/raw_events/universe/<name>/`
  and emits ordered `SeedStep`s.
- `core/reporting.py`: four panels rendered, scenario detail row carries
  difficulty / source_mix / coverage / precision.
- `runner.py`: universe seeding + distractor injection + per-dimension
  scoring.
- `cli.py`: `--difficulty` / `--source-mix` / `--dimension` filters and
  `list --grid` matrix.
- `tools/gen_distractors.py`: author-time noise generator.
- Engine-side: `adapters/outbound/connectors/_bench_stubs.py` —
  `SlackStubConnector`, `RepoDocsStubConnector`, `AlertingStubConnector`,
  `DeployStubConnector`. All four advertise `fetch_capable=False,
  plan_capable=True` and emit a minimal `ReconciliationPlan` per
  envelope. Registered in both `container.py` and
  `standalone_container.py`.
- Canonical Acme seed: `fixtures/raw_events/universe/acme/` —
  architecture, CODEOWNERS-style team roster, ADR-007 / ADR-014 /
  ADR-021, db-pool / deploy-rollback runbooks, baseline deploy event.
- Seed scenarios: `bug_postgres_pool_exhaustion_recurrence` (relocated
  + migrated to v3), `pref_logging_structlog`, `infra_topology_basic`,
  `time_recent_changes_window_14d`, `combo_onboarding_new_engineer`.
- Unit tests: `test_scenario_loader.py` (9), `test_replay_distractors.py`
  (4), `test_evaluators_subaxes.py` (6), `test_reporting_panels.py` (2)
  — **21 / 21 passing**.

---

## 9. Instance tracking table (living)

Status legend: `planned` · `authoring` · `runs` (executes end-to-end) ·
`passing` (meets pass_score) · `regressed` (was passing, now failing).

### 9.1 `PREF` — Project preferences

| ID | Title | Difficulty | Sources | Status |
|---|---|---|---|---|
| `pref_logging_structlog` | Logging conventions surface for new handler | easy | repo_docs + slack | **authored** |
| `pref_error_handling_subclass` | New endpoint must use AcmeError | easy | repo_docs + github | planned |
| `pref_pydantic_over_dataclass` | ADR-014 surfaces when writing a DTO | medium | repo_docs + github | planned |
| `pref_test_layout_conventions` | Test conventions surface for new test | medium | repo_docs + github | planned |
| `pref_naming_and_style` | Style rules surface from accumulated review comments | hard | github (reviews only) | planned |
| `pref_security_input_validation` | Security rules surface for an auth-touching endpoint | hard | notion + slack + repo_docs | planned |

### 9.2 `INFRA` — Project infra & architecture

| ID | Title | Difficulty | Sources | Status |
|---|---|---|---|---|
| `infra_topology_basic` | Service-to-service deps for inventory-svc | easy | repo_docs | **authored** |
| `infra_env_adapter_matrix` | Adapter choice differs prod vs. staging | medium | notion + github | planned |
| `infra_deploy_flow` | Argo + GHA deployment path for checkout-api | medium | deploy + repo_docs | planned |
| `infra_secret_routing` | Where each service reads which secret | hard | repo_docs + slack | planned |
| `infra_oncall_topology` | Owner + on-call mapping per service | hard | repo_docs + slack + notion | planned |

### 9.3 `TIME` — Timeline

| ID | Title | Difficulty | Sources | Status |
|---|---|---|---|---|
| `time_recent_changes_window_14d` | Last 14 days in checkout-api (with out-of-window distractor) | easy | github | **authored** |
| `time_change_attribution_pr_to_issue` | Which PR closed which issue, in order | easy | github + linear | planned |
| `time_change_to_symptom_correlation` | Deploy → Sentry spike within 30 min | medium | deploy + alerting | planned |
| `time_multi_quarter_drift` | Cumulative changes to auth across 90 days | medium | github + linear | planned |
| `time_hotspot_detection` | Files churned > N times in 30 days | hard | github | planned |
| `time_window_edge_cases` | Boundary correctness: events at t=−14d 00:00 | adversarial | github + linear + deploy | planned |

### 9.4 `BUG` — Bug / debug repo

| ID | Title | Difficulty | Sources | Status |
|---|---|---|---|---|
| `bug_postgres_pool_exhaustion_recurrence` | OPS-218 → OPS-389 (existing, migrated to v3 schema) | hard | linear | **authored** (was 12.9 / 100 under v1 schema; rebaseline pending) |
| `bug_postgres_pool_exhaustion_recurrence_v2` | Same, but with Slack + PR + alerting | hard | linear + slack + github + alerting | planned |
| `bug_oom_kafka_consumer` | Memory leak in consumer; prior fix at -120d | easy | linear + github | planned |
| `bug_redis_connection_flap` | Network flake; prior runbook covers it | easy | alerting + repo_docs | planned |
| `bug_dependency_conflict_pydantic_v1_v2` | Version pin drift; surfaces ADR-014 | medium | github + notion | planned |
| `bug_auth_token_expiry_race` | Race condition; prior postmortem | medium | linear + notion + slack | planned |
| `bug_config_drift_staging_vs_prod` | Symptom seen in staging, prod ok | hard | deploy + slack + github | planned |
| `bug_celery_worker_starvation` | Adversarial: two near-misses match the symptom; only one is the real prior | adversarial | linear + slack + github + alerting | planned |

### 9.5 `COMBO` — Composite

| ID | Dimensions | Title | Difficulty | Sources | Status |
|---|---|---|---|---|---|
| `combo_onboarding_new_engineer` | PREF + INFRA | "I'm new to the team — how do I add a Postgres-backed endpoint to inventory-svc?" | medium | repo_docs + github | **authored** |
| `combo_debug_recent_infra_change` | TIME + BUG + INFRA | "Inventory is 503ing. What changed and is it a known pattern?" | hard | all | planned |
| `combo_postmortem_drafting` | BUG + TIME | "Draft a postmortem for OPS-389 from what we know" | hard | linear + github + alerting + slack | planned |
| `combo_release_readiness_check` | TIME + INFRA | "Is checkout-api safe to release today?" | medium | github + deploy + alerting | planned |
| `combo_decision_lookup_with_recency` | PREF + TIME | "Has the logging convention changed in the last 60 days?" | medium | notion + slack + repo_docs | planned |

---

## 10. Open questions

These are the design choices I deferred and want to revisit once P0–P2 land:

1. **Cross-scenario interference.** Should the bench run a "soak" tier where
   all instances seed into one pot and queries run on top of the union, so we
   measure whether bench instance N regresses bench instance M's retrieval?
2. **Reconciliation-agent ablation.** Should the bench have a `--ablate
   reconciliation` mode that bypasses the agent and uses deterministic
   extraction, so we can measure how much value the agent adds per use case?
3. **Real-data corpus.** Out of scope here but plausibly next — a parallel
   `universe/real/` seeded from an anonymised slice of Potpie's own GitHub /
   Linear, so synthetic improvements transfer.
4. **Judge model choice per dimension.** Some criteria (e.g. `correct_chronology`
   in TIME) are mechanically checkable without an LLM. Move those out of the
   judge into deterministic retrieval assertions to cut cost and variance?
