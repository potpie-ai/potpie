# Context Engine Benchmarks

A live, no-mock benchmark harness for the context engine. Every scenario:

1. Creates an **ephemeral pot** on a configured engine.
2. Replays a sequence of **real recorded webhook payloads** (GitHub PRs, Linear issues, ...) through the canonical `/events/reconcile` path so the **real reconciliation agent and ontology classifier** are exercised end-to-end.
3. Snapshots the resulting graph and asserts **ingestion / ontology quality** (right entities, right edges, no soft-downgrades).
4. Issues a `context_resolve` call and asserts **retrieval correctness** (cited the right events, used the right include keys).
5. Hands the response to an **Opus LLM judge** that scores synthesis quality against a per-scenario rubric.
6. Drops the pot.

## Why no mocks

The previous benchmark mocked the reconciliation agent and used canned `MockIntelligenceProvider` responses, which is exactly what the headline 81 % score was measuring. The redesign rejects that: reconciliation is the engine's product, so the bench drives it for real.

## Layout

```
benchmarks/
├── README.md
├── cli.py                          # python -m benchmarks {run,list,fixture,report}
├── core/                           # lifecycle, ingestion, replay, query, graph_inspect, scenario, result, reporting
├── evaluators/                     # ingestion_quality, retrieval, llm_judge
├── fixtures/
│   ├── raw_events/                 # Recorded + redacted webhook payloads
│   │   ├── github/
│   │   └── linear/
│   └── README.md                   # How to capture + redact
├── use_cases/
│   ├── feature/scenarios/*.yaml
│   ├── debugging/scenarios/*.yaml
│   ├── review/scenarios/*.yaml
│   ├── operations/scenarios/*.yaml
│   └── onboarding/scenarios/*.yaml
└── reports/                        # gitignored output
```

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
python -m benchmarks run                                # all quick-tier scenarios
python -m benchmarks run --use-case debugging           # filter by use case
python -m benchmarks run --scenario debug_recurring_neo4j_pool_exhaustion
python -m benchmarks run --tier extended                # nightly; not run by default
python -m benchmarks run --baseline reports/main.json   # regression check vs. previous report

python -m benchmarks list                               # show scenarios + tier + use_case
python -m benchmarks fixture validate                   # lint all raw_events JSON
python -m benchmarks report reports/latest.json --format markdown
```

## Authoring a new scenario

1. Pick a use case (`feature`, `debugging`, `review`, `operations`, `onboarding`). If a new one is needed, add a sibling directory under `use_cases/`.
2. Capture or reuse the **raw webhook payloads** the scenario depends on. See `fixtures/README.md` for the capture + redaction workflow.
3. Write a `<id>.yaml` under `use_cases/<use_case>/scenarios/`. The schema is:

```yaml
id: <unique-snake-case-id>
use_case: feature | debugging | review | operations | onboarding
tier: quick | extended
description: |
  One-paragraph statement of the agent task and why this is hard.

ingest:                              # Replayed in order, real reconciliation each time
  - { event: <connector>/<file>.json, at: "-60d" }   # at: ISO8601 or relative-to-now offset
  - ...

post_ingest_assertions:              # Evaluates reconciliation agent + ontology
  graph_must_contain_entities:
    - { label: <Label>, key_pattern: <regex>, where: { <prop>: <value> }, min_count: 1 }
  graph_must_contain_edges:
    - { from_label: <Label>, to_label: <Label>, type: <EDGE_TYPE> }
  no_orphan_entities: true
  reconciliation:
    soft_downgrades_max: 0
    failed_events_max: 0

query:
  intent: <intent>
  scope: { <key>: <value> }
  include: [<include_key>, ...]
  mode: fast | balanced | verify | deep
  source_policy: references_only | summary | verify | snippets

retrieval_assertions:                # Cheap deterministic checks
  required_includes_used: [<include_key>, ...]
  source_refs_min: <int>
  must_cite_event_id: <connector>/<file>.json
  forbid_in_answer: ["..."]

judge:                               # Opus rubric
  pass_score: 75                     # 0..100
  criteria:
    - { name: <slug>, weight: <int>, pass_threshold: 1..5,
        prompt: "Question the judge will answer about the response." }
```

Every scenario is graded on three axes (ingestion / retrieval / synthesis). The aggregate scenario score is the weighted mean (default 30 / 30 / 40). Override with a top-level `axis_weights:` block if needed.

## Tiers

- **quick** — runs on every PR. Target ~20 scenarios total. Each scenario should complete in < 90 s.
- **extended** — runs nightly or before a release. No hard cap; expect minutes-to-hours.

## Reports

Reports are JSON-first (`reports/<timestamp>.json`) with a `--format markdown` view for humans. A baseline diff highlights regressions per axis (ingestion, retrieval, synthesis) so a small synthesis dip doesn't get hidden by a stable retrieval score.
