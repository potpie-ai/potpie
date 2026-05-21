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
├── cli.py                       # python -m benchmarks {run,list,fixture,report}
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

The engine ships production-grade readers for **GitHub** and **Linear**, and a minimal **Notion** reader. The four other connector kinds the bench uses — **Slack**, **Repo Docs**, **Alerting**, **Deploy** — are wired up as *passive stub connectors* (see `app/src/context-engine/adapters/outbound/connectors/_bench_stubs.py`). They emit a minimal `ReconciliationPlan` per envelope (one canonical entity, no edges) and let the reconciliation agent do the rest. The contract for swapping in a production reader later is the same `SourceConnectorPort` they implement.

## Configuration

The bench talks to a real engine. Required environment:

| Variable | Purpose |
|---|---|
| `POTPIE_BENCH_API_URL` | Engine base URL (e.g. `http://127.0.0.1:8001`). Falls back to `POTPIE_API_URL`. |
| `POTPIE_BENCH_API_KEY` | API key for an account that can create pots. Falls back to `POTPIE_API_KEY`. |
| `OPENAI_API_KEY` | Used by the LLM judge (default model `gpt-5.4`). |
| `POTPIE_BENCH_JUDGE_MODEL` | Optional. Override the judge model. Default `gpt-5.4`. |
| `POTPIE_BENCH_REPO` | Optional. `owner/repo` to attach to each ephemeral pot. Default `acme/sandbox`. |
| `CONTEXT_ENGINE_RECONCILIATION_MODEL` | Optional. Reconciliation-agent model on the engine side. Default `openai-responses:gpt-5.4-mini`. |

## Running

```bash
# --- Pre-flight (do these before any long run) ---
python -m benchmarks smoke                                # in-process pipeline check (<1 s, no engine)
python -m benchmarks probe                                # engine + connector + drain check (~15 s)

# --- The bench ---
python -m benchmarks run                                  # all quick-tier scenarios
python -m benchmarks run --use-case BUG                   # filter by knowledge dimension
python -m benchmarks run --difficulty hard                # only hard scenarios
python -m benchmarks run --source-mix full                # multi-source scenarios
python -m benchmarks run --dimension TIME                 # includes composites touching TIME
python -m benchmarks run --scenario bug_postgres_pool_exhaustion_recurrence
python -m benchmarks run --tier extended                  # nightly; not run by default
python -m benchmarks run --concurrency 4                  # 4 scenarios in parallel (pots are isolated)

# --- Inspection + diff ---
python -m benchmarks list                                 # flat list
python -m benchmarks list --grid                          # use_case × difficulty coverage matrix
python -m benchmarks fixture validate                     # lint all raw_events JSON
python -m benchmarks report reports/latest.json --format markdown
python -m benchmarks report reports/latest.json --baseline reports/main.json   # diff (non-zero exit on regression)

# --- Authoring ---
python -m benchmarks.tools.new_scenario --use-case PREF --difficulty medium --id pref_xxx
python -m benchmarks.tools.gen_distractors --template <fixture> --count 25 --id-prefix noise_x --out fixtures/raw_events/noise/
```

## Operational commands explained

- **`smoke`** — runs every scenario through the bench-side pipeline (fixture resolution, evaluator math, reporting) against synthetic snapshots. No engine required. Sub-second. The right gate on every PR — catches harness regressions before the long run.
- **`probe`** — pre-flight against the live engine. Confirms `/context/status` returns the expected connector kinds, then submits one minimal envelope per kind and waits ~15 s for it to leave `queued`. If reconciliation is dead this command finishes in seconds with a clear diagnostic instead of 11 minutes of timeouts mid-run.
- **`run --concurrency N`** — pots are isolated by construction, so scenarios run in parallel safely. Within each scenario, seed + distractor events also run in parallel (a 6-worker pool) — only signal events stay sequential because they encode arrival order.

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
python -m benchmarks.tools.gen_distractors \
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
