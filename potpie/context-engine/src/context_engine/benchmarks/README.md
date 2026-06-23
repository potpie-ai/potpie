# Context Engine Benchmarks

A live, no-mock benchmark harness for the context engine. Every scenario:

1. Creates an **ephemeral pot** on a configured engine.
2. (If declared) loads the **canonical Acme universe seed** — services, team, ADRs, runbooks — through the same `/events/reconcile` path scenarios use.
3. Replays a sequence of **real recorded webhook payloads** (GitHub PRs, Linear issues, Slack messages, repo docs, alerts, deploys) plus any **distractor events** declared by the scenario, interleaved on the ingestion timeline.
4. Snapshots the resulting graph and asserts **ingestion / ontology quality** (right entities, right edges, no soft-downgrades, and — new — that the graph stays *clean* of distractor entities the scenario explicitly forbids).
5. Issues a `context_resolve` call and asserts **retrieval correctness** (cited the right events, used the right include keys, did not cite distractor events, and — for TIME scenarios — stayed inside the declared window).
6. Hands the response to an **Opus LLM judge** that scores synthesis quality against a per-scenario rubric, with optional per-dimension attribution for composite scenarios.
7. Drops the pot.

The full design lives in `docs/context-graph/bench-plan.md`. This file documents how to *run* the bench; the plan documents what it *measures*.

## Why no mocks

The previous benchmark mocked the reconciliation agent and used canned `MockIntelligenceProvider` responses, which is exactly what the headline 81 % score was measuring. The redesign rejects that: reconciliation is the engine's product, so the bench drives it for real.

## Use-case taxonomy (v3, 2026-05-20)

Four **knowledge dimensions** + a composite modifier:

| Code | What it tests | Default weights (ing / ret / syn) |
|---|---|---|
| `PREF`  | Project preferences — conventions, ADRs, framework choices | 20 / 40 / 40 |
| `INFRA` | Project infra / architecture — topology, owners, environments | 30 / 40 / 30 |
| `TIME`  | Timeline — recent changes, chronology, change-to-symptom links | 40 / 30 / 30 |
| `BUG`   | Bug repo — prior incidents, fixes, recurrence detection | 25 / 35 / 40 |
| `COMBO` | Composite scenarios that exercise ≥ 2 of the above | 30 / 35 / 35 |

`COMBO` scenarios declare `dimensions: [PREF, INFRA]` (etc.); the report breaks their score out per dimension.

## Layout

```
benchmarks/
├── README.md                    # this file
├── cli.py                       # python -m context_engine.benchmarks {run,list,fixture,report}
├── core/
│   ├── lifecycle.py             # ephemeral pot create/reset
│   ├── ingestion.py             # /events/reconcile + ledger poll
│   ├── replay.py                # envelope load, distractor expansion, timeline assembly
│   ├── query.py                 # context_resolve client
│   ├── graph_inspect.py         # post-ingest graph snapshot
│   ├── scenario.py              # v3 schema + loader
│   ├── universe.py              # canonical-universe seed loader
│   ├── result.py                # AxisScore + sub-axes + by_dimension
│   └── reporting.py             # by_use_case / by_dimension / by_difficulty / by_source_mix
├── evaluators/
│   ├── coverage.py              # recall sub-axis
│   ├── precision.py             # purity sub-axis
│   ├── ingestion_quality.py     # primary + sub-axes for ingestion
│   ├── retrieval.py             # primary + sub-axes for retrieval
│   └── llm_judge.py             # synthesis judge + per-dimension breakout
├── tools/
│   └── gen_distractors.py       # author-time noise generator
├── fixtures/
│   ├── raw_events/              # Recorded + redacted webhook envelopes
│   │   ├── github/  linear/  slack/  notion/  repo_docs/  alerting/  deploy/
│   │   └── universe/acme/       # canonical seed bundle
│   └── README.md                # how to capture + redact
├── use_cases/
│   ├── PREF/scenarios/*.yaml
│   ├── INFRA/scenarios/*.yaml
│   ├── TIME/scenarios/*.yaml
│   ├── BUG/scenarios/*.yaml
│   └── COMBO/scenarios/*.yaml
└── reports/                     # gitignored output
```

## Multi-source ingestion

The engine ships production-grade readers for **GitHub** and **Linear**, and a minimal **Notion** reader. The four other connector kinds the bench uses — **Slack**, **Repo Docs**, **Alerting**, **Deploy** — are wired up as *passive stub connectors* (see `potpie/context-engine/adapters/outbound/connectors/_bench_stubs.py`). They emit a minimal `ReconciliationPlan` per envelope (one canonical entity, no edges) and let the reconciliation agent do the rest. The contract for swapping in a production reader later is the same `SourceConnectorPort` they implement.

## Two ways to run the engine

The bench can drive the engine two ways:

1. **HTTP (default).** Talks to a running engine over `POTPIE_BENCH_API_URL`
   — which is the app's own API port (`:8001`). Needs gunicorn on 8001 + a
   Celery worker on the `context-graph-etl` queue.
2. **In-process (`--local` / `POTPIE_BENCH_INPROCESS=1`).** Builds the engine
   container in-process and reconciles **inline** against the shared
   Postgres/Neo4j — **no HTTP server on :8001 and no Celery worker required**.
   This keeps the bench non-blocking for the rest of the app (the app keeps
   8001) and is fully self-contained. Ingestion is single-threaded and
   batched (submit-all → reconcile-once → read-back), which also sidesteps the
   threads-pool *"Event loop is closed"* reconciliation race. Implemented in
   `core/local_engine.py` (`InProcessEngineClient`).

```bash
# In-process: needs only Postgres + Neo4j up (e.g. `make infra-up`) + OPENAI_API_KEY.
python -m context_engine.benchmarks run --local --tier quick --ingest-timeout 600
python -m context_engine.benchmarks run --local --scenario infra_topology_basic --skip-judge
```

## Configuration

For the **HTTP** path the bench talks to a real engine. Required environment:

| Variable | Purpose |
|---|---|
| `POTPIE_BENCH_INPROCESS` | `1` to run the engine in-process (no :8001 server, no worker). Same as `run --local`. |
| `POTPIE_BENCH_API_URL` | (HTTP path) Engine base URL (e.g. `http://127.0.0.1:8001`). Falls back to `POTPIE_API_URL`. |
| `POTPIE_BENCH_API_KEY` | (HTTP path) API key for an account that can create pots. Falls back to `POTPIE_API_KEY`. |
| `OPENAI_API_KEY` | Used by the LLM judge (default model `gpt-5.4`). |
| `POTPIE_BENCH_JUDGE_MODEL` | Optional. Override the judge model. Default `gpt-5.4`. |
| `POTPIE_BENCH_REPO` | Optional. `owner/repo` to attach to each ephemeral pot. Default `acme/sandbox`. |
| `CONTEXT_ENGINE_RECONCILIATION_MODEL` | Optional. Reconciliation-agent model on the engine side. Default `openai-responses:gpt-5.4-mini`. |

## Running

```bash
# --- Pre-flight (do these before any long run) ---
python -m context_engine.benchmarks smoke                                # in-process pipeline check (<1 s, no engine)
python -m context_engine.benchmarks probe                                # engine + connector + drain check (~15 s)

# --- The bench ---
python -m context_engine.benchmarks run                                  # all quick-tier scenarios
python -m context_engine.benchmarks run-light                            # 5 scenarios, 5-way parallel, invariant judge (see below)
python -m context_engine.benchmarks run-light --local                    # same, in-process (no :8001 / no worker)
python -m context_engine.benchmarks run --use-case BUG                   # filter by knowledge dimension
python -m context_engine.benchmarks run --difficulty hard                # only hard scenarios
python -m context_engine.benchmarks run --source-mix full                # multi-source scenarios
python -m context_engine.benchmarks run --dimension TIME                 # includes composites touching TIME
python -m context_engine.benchmarks run --scenario bug_postgres_pool_exhaustion_recurrence
python -m context_engine.benchmarks run --invariant                      # schema-independent judge instead of per-rubric (any filter)
python -m context_engine.benchmarks run --tier extended                  # nightly; not run by default
python -m context_engine.benchmarks run --concurrency 4                  # 4 scenarios in parallel (pots are isolated)
python -m context_engine.benchmarks run --local --concurrency 4          # in-process, 4 worker processes
python -m context_engine.benchmarks run -v                                # verbose: also show engine/HTTP logs

# --- Inspection + diff ---
python -m context_engine.benchmarks list                                 # flat list
python -m context_engine.benchmarks list --grid                          # use_case × difficulty coverage matrix
python -m context_engine.benchmarks fixture validate                     # lint all raw_events JSON
python -m context_engine.benchmarks report reports/latest.json --format markdown
python -m context_engine.benchmarks report reports/latest.json --baseline reports/main.json   # diff (non-zero exit on regression)

# --- Authoring ---
python -m context_engine.benchmarks.tools.new_scenario --use-case PREF --difficulty medium --id pref_xxx
python -m context_engine.benchmarks.tools.gen_distractors --template <fixture> --count 25 --id-prefix noise_x --out fixtures/raw_events/noise/
```

## Operational commands explained

- **`smoke`** — runs every scenario through the bench-side pipeline (fixture resolution, evaluator math, reporting) against synthetic snapshots. No engine required. Sub-second. The right gate on every PR — catches harness regressions before the long run.
- **`probe`** — pre-flight against the live engine. Confirms `/context/status` returns the expected connector kinds, then submits one minimal envelope per kind and waits ~15 s for it to leave `queued`. If reconciliation is dead this command finishes in seconds with a clear diagnostic instead of 11 minutes of timeouts mid-run.
- **`run --concurrency N`** — pots are isolated by construction, so scenarios run in parallel safely. See "Concurrency & progress" below for how the two engine modes parallelize differently.
- **`run-light`** — curated 5-scenario subset (one per dimension: PREF / INFRA / TIME / BUG / COMBO), 5-way parallel, **invariant** judging on. The fastest end-to-end signal we have. Use it as the smoke test before a full `run`.

## Schema-independent (invariant) grading

The default judge grades the agent's answer against a per-scenario rubric whose criteria name specific fixture ids (OPS-218, ADR-021), and the deterministic axes grade against the engine's graph shape and the include vocabulary it currently happens to expose. Both move when the ontology / read trunk changes, even when the agent's answer is just as good.

`run --invariant` (and the default `run-light`) replaces that with a single judge call that grades the answer against **only the scenario's input signal events** — no rubric, no fixture ids, no include vocab. Four 0..100 sub-scores:

| Score | Question |
|---|---|
| `faithfulness` | Every concrete claim in the answer is supported by the events. Hallucinated identifiers / people / fixes collapse this score. |
| `coverage`     | The answer surfaces the facts a careful reader of the input events would consider essential to the question. |
| `clarity`      | The answer is structured so a working engineer can understand what happened in seconds. |
| `usefulness`   | The answer concretely helps the user do whatever the question implies (debug, decide, follow a convention, plan a change). |

Aggregate: weighted mean (default 30/30/20/20 — faithfulness + coverage carry 60 %, so cosmetic gains can't paper over hallucinations or omissions). Pass score: 65.

In invariant mode the ingestion + retrieval axes still run end-to-end (the engine has to actually ingest the events and answer the query) but they don't gate pass/fail and they don't enter the aggregate. They stay in the JSON report as diagnostics so you can still see e.g. "the engine is now ignoring this include key" alongside the answer-level score.

The judge is configurable via `POTPIE_BENCH_INVARIANT_JUDGE_MODEL` (falls back to `POTPIE_BENCH_JUDGE_MODEL`, default `gpt-5.4`).

## Concurrency & progress

**Progress logging.** A run prints a start banner (mode / concurrency / tier / judge), a `[i/N]` line per scenario with phase breadcrumbs (events ingested → reconcile time → graph snapshot size → retrieval → synthesis), a live `passed k/done` tally, and an end summary (total time, per-scenario pass/fail, slowest scenarios). `benchmarks.*` logs show by default; `-v` / `--verbose` additionally surfaces the chatty engine + HTTP loggers.

**Concurrency model** (`--concurrency N`):

| Engine mode | How `--concurrency N` parallelizes | Why |
|---|---|---|
| **HTTP** (default) | N **threads** fan scenarios out; each thread submits + polls over HTTP while reconciliation runs in the engine's Celery worker. Within a scenario, seeds + distractors submit via a 6-worker pool; signals stay ordered. | HTTP calls are thread-safe; the worker does the heavy LLM work. Throughput is bounded by the worker's pool (run it `--pool=solo` to dodge the agent's threads race). |
| **In-process** (`--local`) | N **processes** (one scenario per subprocess). Within a scenario, ingestion is single-threaded (submit-all → reconcile-once → read-back). | The reconciliation agent does a per-batch `asyncio.run`; sharing a process across scenarios (threads) hits an "Event loop is closed" race. Separate processes give each scenario its own loop / agent / DB session and genuinely parallelize the LLM-bound reconciliation. |

**Cost note.** The dominant per-scenario cost is the reconciliation agent — one LLM run over the scenario's events. The 12-event Acme universe is re-reconciled per scenario (e.g. ~4–5 min for a ~19-event scenario), and `hard`/`adversarial` scenarios add heavy distractors. So: prefer `--concurrency` (≈ linear speedup, bounded by CPU/LLM rate limits), keep the judge off (`--skip-judge`) for cheap dry runs, and dial distractor `count:` down if a run is too slow. (A future `--reuse-pot` that seeds the universe once is the biggest remaining win — see bench-plan §10.)

## Authoring a new scenario

1. Pick a use case (`PREF`, `INFRA`, `TIME`, `BUG`, or `COMBO`).
2. Capture or reuse the **raw event envelopes** the scenario depends on. See `fixtures/README.md`.
3. Write a `<id>.yaml` under `use_cases/<USE_CASE>/scenarios/`. Schema (everything past `judge:` is unchanged from v1; everything above is v3):

```yaml
id: <unique-snake-case-id>
use_case: PREF | INFRA | TIME | BUG | COMBO
dimensions: [PREF, INFRA]                  # required for COMBO; ≥ 2 entries
tier: quick | extended
difficulty: easy | medium | hard | adversarial
source_mix: single | dual | full | adversarial

universe: acme                             # optional — seed the canonical Acme universe
seed:                                      # optional — pin individual seed events
  - { event: universe/acme/00-architecture.json, at: "-365d" }

description: |
  One-paragraph statement of the agent task and why this is hard.

ingest:                                    # Signal events, replayed in order
  - { event: github/pr_merge__1042.json, at: "-14d", tags: [signal] }

distractor_events:                         # Optional — noise interleaved with signals
  - { event: github/pr_merge__noise_*.json, at: "-21d..-7d", count: 12, shape: "noise/random" }

post_ingest_assertions:
  graph_must_contain_entities:
    - { label: <Label>, key_pattern: <regex>, where: { <prop>: <value> }, min_count: 1 }
  graph_must_contain_edges:
    - { from_label: <Label>, to_label: <Label>, type: <EDGE_TYPE> }
  graph_must_not_contain:                  # Negative class — feeds the precision sub-axis
    - { label: Issue, key_pattern: "NOISE-.*", max_count: 0 }
  no_orphan_entities: false
  reconciliation:
    soft_downgrades_max: 0
    failed_events_max: 0

query:
  intent: <intent>
  scope: { <key>: <value> }
  include: [<include_key>, ...]
  mode: fast | balanced | verify | deep
  source_policy: references_only | summary | verify | snippets

retrieval_assertions:
  required_includes_used: [<include_key>, ...]
  source_refs_min: <int>
  must_cite_event_id: [<connector>/<file>.json, ...]      # list (or single string)
  must_not_cite_event_id: [<connector>/<noise>.json, ...] # negative class
  forbid_in_answer: ["..."]
  temporal:                                # TIME scenarios only
    must_order_correctly: true
    window: { from: "-14d", to: "0d" }
    out_of_window_refs_max: 0

judge:
  pass_score: 70                            # optional; default picks by difficulty
                                            #   (easy 75 / medium 65 / hard 55 / adversarial 45)
  criteria:
    - name: <slug>
      weight: <int>
      pass_threshold: 1..5
      dimensions: [TIME, BUG]              # optional — for COMBO per-dimension breakout
      prompt: "Question the judge will answer about the response."
```

### Opt-in sub-axis gates

Coverage and precision are reported as separate numbers on every scenario, but they do not block pass/fail by default. Scenarios that *care* about a specific recall or purity floor can declare it:

```yaml
post_ingest_assertions:
  coverage_floor: 80      # ingestion axis fails if <80% of assertions match
  precision_floor: 90     # ingestion axis fails if forbidden entities slip in
  # ... structural assertions ...

retrieval_assertions:
  coverage_floor: 100     # every must_cite event must actually be cited
  precision_floor: 80     # forbidden citations cost real points
  # ... must_cite_event_id etc. ...
```

Without these fields, sub-axes are diagnostic-only and never gate.

Every scenario is graded on three axes (ingestion / retrieval / synthesis) plus two sub-axes (coverage / precision). The aggregate scenario score is the weighted mean of the primary axes; weights come from the use case unless `axis_weights:` overrides.

## Distractor generation

Hand-authoring 25:1 distractor ratios is busywork. Use the generator:

```bash
python -m context_engine.benchmarks.tools.gen_distractors \
    --template fixtures/raw_events/github/pr_merged__998__inventory_unrelated.json \
    --count 25 \
    --id-prefix noise_inventory_pr \
    --out fixtures/raw_events/noise/inventory/
```

Then reference the output directory in the scenario via a glob:

```yaml
distractor_events:
  - { event: "noise/inventory/noise_inventory_pr__*.json", at: "-21d..-7d", count: 25 }
```

## Tiers

- **quick** — runs on every PR. Target 30 scenarios total. Each scenario should complete in < 90 s (excluding the LLM judge, which is opt-in via `--skip-judge` for cheap dry runs).
- **extended** — runs nightly or before a release. No hard cap; expect minutes-to-hours. Target 60 scenarios.

## Reports

Reports are JSON-first (`reports/<timestamp>.json`) with a `--format markdown` view for humans. Four aggregation panels appear in every report:

- **By use case** — count, aggregate, ingestion / retrieval / synthesis, coverage, precision.
- **By dimension** — composite scenarios decomposed across PREF / INFRA / TIME / BUG.
- **By difficulty** — easy / medium / hard / adversarial curve.
- **By source mix** — single / dual / full / adversarial curve.

A baseline diff (`--baseline reports/main.json`, planned) highlights regressions per panel cell so a small synthesis dip on `hard` doesn't get hidden by a stable easy-tier retrieval score.
