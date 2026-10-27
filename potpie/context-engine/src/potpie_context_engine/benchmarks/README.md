# Context Engine Benchmark Utilities

This package contains scenario loading, fixture validation and redaction,
synthetic smoke evaluation, evidence scoring, and report comparison utilities.
The live `run` and `run-light` commands and application-specific in-process
engine driver have been removed. They depended on retired application wiring
and an unsupported query endpoint.

Canonical read helpers accept an explicit `GraphService`: `resolve_context`
uses `ResolveRequest`, and `snapshot_graph` reconstructs a bounded sample from
returned claim evidence. The sample is limited to 200 ranked items and is not a
complete graph export. A new live ingestion benchmark must supply its runtime
and ingestion lifecycle explicitly; this package does not currently provide
an end-to-end live runner.

## Commands

Run these from an environment with `potpie-context-engine` installed:

```bash
python -m potpie_context_engine.benchmarks list --grid
python -m potpie_context_engine.benchmarks smoke
python -m potpie_context_engine.benchmarks fixture validate
python -m potpie_context_engine.benchmarks fixture redact path/to/event.json
python -m potpie_context_engine.benchmarks report reports/latest.json --format markdown
python -m potpie_context_engine.benchmarks report reports/latest.json --baseline reports/main.json
```

`smoke` validates available fixtures and exercises evaluators using synthetic
empty evidence; it does not validate a running engine. Scenario YAML files are
shipped, but the complete raw-event corpus must be supplied separately.
`report` compares existing results and exits nonzero when the configured
regression threshold is exceeded.

The optional `probe` command checks a configured ingestion server's health,
connector registration, and event completion. It requires
`POTPIE_BENCH_API_URL` and `POTPIE_BENCH_API_KEY` (or `POTPIE_API_URL` and
`POTPIE_API_KEY`) and creates temporary pots for its checks.

## Evaluation model

Scenarios describe project preferences (`PREF`), infrastructure (`INFRA`),
timelines (`TIME`), prior bugs (`BUG`), or composites (`COMBO`). Composite cases
name their dimensions. Difficulty and source mix remain independent report
axes; a `light: true` tag identifies the curated five-scenario subset.

Deterministic evaluators score structural coverage, retrieval, citations, and
precision. The invariant LLM evaluator grades an answer against supplied input
events using faithfulness, coverage, clarity, and usefulness. It is an explicit
library call and requires a configured model client. Report utilities retain
per-dimension results and baseline comparisons.

## Authoring a new scenario

1. Pick a use case (`PREF`, `INFRA`, `TIME`, `BUG`, or `COMBO`).
2. Capture or reuse the **raw event envelopes** the scenario depends on. See `fixtures/README.md`.
3. Write a `<id>.yaml` under `use_cases/<USE_CASE>/scenarios/`. Scenario schema:

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
python -m potpie_context_engine.benchmarks.tools.gen_distractors \
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

`quick` and `extended` are scenario metadata used to select evaluation inputs.
They do not imply a scheduled or available live execution mode.

## Reports

Reports are JSON-first (`reports/<timestamp>.json`) with a `--format markdown` view for humans. Four aggregation panels appear in every report:

- **By use case** — count, aggregate, ingestion / retrieval / synthesis, coverage, precision.
- **By dimension** — composite scenarios decomposed across PREF / INFRA / TIME / BUG.
- **By difficulty** — easy / medium / hard / adversarial curve.
- **By source mix** — single / dual / full / adversarial curve.

A baseline diff (`--baseline reports/main.json`, planned) highlights regressions per panel cell so a small synthesis dip on `hard` doesn't get hidden by a stable easy-tier retrieval score.
