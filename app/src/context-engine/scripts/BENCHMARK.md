# Context Graph Benchmark

A comprehensive benchmark harness for the Potpie context graph, implemented as the `benchmarks` package under `app/src/context-engine/benchmarks/`.

## Modes

| Mode | Description | Use When |
|------|-------------|----------|
| `mock` | In-process mock intelligence provider | Fastest baseline; no I/O |
| `http-e2e` | In-process FastAPI test client | Router-level latency without network |
| `api` | Live Potpie server at `POTPIE_API_URL` | Real-world end-to-end benchmarking |

## Quick Start

```bash
# From the context-engine package root
cd app/src/context-engine

# Mock mode (fastest, no server)
uv run python -m benchmarks.cli mock

# HTTP-E2E mode (in-process router)
uv run python -m benchmarks.cli http-e2e

# API mode (live server; needs POTPIE_API_KEY)
uv run python -m benchmarks.cli api

# Or via the convenience wrapper
uv run python scripts/benchmark_context_engine.py mock
```

## Scenarios

The benchmark exercises the full context graph surface:

### Seeding
- **Episodes** — seed episodic memory (decisions, workflows, debugging, runbooks, preferences)
- **Records** — seed structured records (decision, fix, workflow)
- **PR bundles** — ingest pull request fixtures as episodes (or live PRs with `--ingest-pr-live`)

### Query Scenarios
- **Agent context recipes** — `feature`, `debugging`, `review`, `operations` intents
- **Semantic search** — semantic retrieval with varying limits
- **Readiness / quality checks** — status and recipe validation
- **Complex synthesis** (`scenarios_complex.json`) — cross-domain queries that span ≥3 evidence families: causal reasoning, cross-PR coordination, impact analysis, risk assessment, policy compliance.
- **Timeline** (`scenarios_timeline.json`) — recent-pulse queries, named periods, user/branch/file scopes.
- **v2 context management** (`scenarios_v2_context.json`) — exercises the v2 ontology entities and the new assertion types:
  - `v2_active_initiatives` / `v2_active_risks` / `v2_open_questions` — surfacing new intent entities
  - `v2_active_migration` / `v2_feature_flags_gating` — operational state entities
  - `v2_policies_for_agents` — policy compliance under the merged Policy entity
  - `v2_before_you_edit_workflow` — holistic state assembly for an agent about to edit code
  - `v2_negative_space_missing_runbook` — honest gap declaration when data is missing
  - `v2_temporal_current_datastore` — current state vs. superseded (Mongo → Postgres migration)
  - `v2_conflict_surfacing` — `response.conflicts` populated when graph contradictions exist
  - `v2_evidence_strength_distribution` — `evidence_strength` ratio on returned facts

Each scenario carries weighted assertions. Score starts at 100 and each missed assertion subtracts its weight.

### Core assertions

| Field | Weight | Description |
|---|---|---|
| `required_coverage` (per family) | 25 | A coverage family the agent explicitly asked for must be available. Most critical signal. |
| `must_contain` (per token) | 15 | Token must appear in flattened response JSON. |
| `must_not_contain` (per token) | 10 | Hallucination guard — token must not appear anywhere in response. |
| `min_source_refs` | 8 | At least N source references attached to facts. |
| `required_paths` (per path) | 5 | Dotted JSON path must exist and be non-empty. |
| `coverage` (per key) | 5 | Exact match on `response.coverage.<key>`. |
| `quality` (per key) | 5 | Exact match on `response.quality.<key>`. |
| `max_fallbacks` | 3 | At most N fallback entries. |
| `min_facts` | 3 | At least N facts. |
| `max_latency_ms` | 3 | Response latency within budget. |

### Context-management assertions (v2 — phase-9)

These extend the benchmark to evaluate ontology utilization, evidence trust, conflict handling, source-policy compliance, and negative-space honesty.

| Field | Weight | Description |
|---|---|---|
| `expected_entity_labels` (per label) | 15 | At least one record in the response carries this canonical ontology label (`kind` / `label` / `canonical_type` / `labels`, or implicit via bucket key). |
| `expected_edge_types` (per type) | 10 | At least one edge of the given canonical type is present in `facts` / `relations`. Verifies cross-domain joins are intact. |
| `min_evidence_strength_ratio` | 10 | Of records that declare `evidence_strength`, the share tagged `deterministic` or `attested` must meet the threshold. Filters out hypothesis-heavy answers. |
| `expected_conflicts` (per pattern) | 15 | `response.conflicts` contains an entry whose flattened form includes the pattern. Verifies contradicting facts surface, not silently average out. |
| `expected_fallback_reasons` (per reason) | 8 | **Positive credit** — the named fallback reason (`missing_data`, `missing_scope`, `permission_denied`, …) appears in `response.fallbacks`. Rewards honestly declaring gaps over confabulating. |
| `max_facts` | 5 | Cap on returned facts when scope is narrow. Context-window discipline. |
| `forbidden_in_answer` (per token) | 12 | Token must not appear in the synthesized `answer` text. Narrower than `must_not_contain`; ignores metadata mentions. |
| `expected_includes_used` | 6 | `response.coverage.available` must be a superset of the named keys. Catches under-fetch. |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `benchmarks/data/context_graph_benchmark_dataset.json` | Dataset JSON path |
| `--report` | `.tmp/context-graph-benchmark-report.json` | Output path for JSON report |
| `--baseline` | — | Previous report for regression checks |
| `--pot-id` | auto | Pot UUID. API mode auto-detects from server list if omitted |
| `--repo-name` | — | Scope filter for queries and seeding |
| `--iterations` / `-i` | 3 | Iterations per scenario |
| `--concurrency` / `-c` | 4 | Max concurrent scenario requests |
| `--no-seed` | — | Skip fixture seeding |
| `--ingest-pr-live` | — | Call `/ingest-pr` for PR fixtures instead of seeding episodes |
| `--print-json` | — | Print full JSON report to stdout |

## Example: Compare API vs Mock

```bash
uv run python -m benchmarks.cli api  --iterations 5 --report .tmp/bench-api.json
uv run python -m benchmarks.cli mock --iterations 5 --report .tmp/bench-mock.json

# Diff key metrics
uv run python -c "
import json
for path, label in [('.tmp/bench-api.json','API'),('.tmp/bench-mock.json','Mock')]:
    d = json.load(open(path))
    print(f'--- {label} ---')
    for s in d['scenarios']:
        lat = s.get('latency_ms',{})
        ratio = s['score']/s['max_score'] if s['max_score'] else 0
        print(f\"{s['id']:<35} score={ratio:>6.1%} p95={lat.get('p95'):>8.2f}ms errs={s['errors']}\")
"
```

## Interpreting Results

The report JSON contains:

```json
{
  "dataset": {"name": "...", "version": "..."},
  "target": {"mode": "api", "pot_id": "...", "iterations": 3, "concurrency": 4},
  "summary": {
    "ok": true,
    "score": 0.9184,
    "grade": "excellent",
    "scenario_count": 7,
    "error_count": 0,
    "pass_score": 0.72
  },
  "scenarios": [
    {
      "id": "agent_feature_recipe",
      "feature": "agent_context_port",
      "grade": "excellent",
      "score": 5.5,
      "max_score": 5.5,
      "errors": 0,
      "latency_ms": {"min": 120, "p50": 145, "p95": 180, "max": 200},
      "assertions": [...],
      "response_summary": {...}
    }
  ],
  "regressions": []
}
```

### Grades

| Grade | Threshold | Meaning |
|-------|-----------|---------|
| `excellent` | ≥ 90 % | All critical assertions pass |
| `good` | ≥ 75 % | Acceptable for production |
| `watch` | ≥ 60 % | Needs attention |
| `regressed` | < 60 % | Blocking issue |

### Regressions

When `--baseline` is provided, the benchmark flags:
- **score_drop** — scenario score dropped more than `thresholds.max_score_drop` (default 5 %)
- **latency_p95** — p95 latency grew more than `thresholds.max_p95_latency_ratio` (default 3×)

## Writing a new scenario

Scenarios live under `benchmarks/data/scenarios_*.json` and are picked up by the manifest. The minimum shape:

```json
{
  "id": "v2_my_check",
  "feature": "what_this_measures",
  "intent": "feature",
  "tags": ["v2", "policy"],
  "request": {
    "goal": "answer",
    "strategy": "hybrid",
    "query": "...",
    "intent": "feature",
    "include": ["policies", "decisions"],
    "source_policy": "references_only",
    "budget": {"max_items": 8, "timeout_ms": 5000}
  },
  "expected": {
    "must_contain": ["postgres"],
    "expected_entity_labels": ["Policy"],
    "expected_edge_types": ["APPLIES_TO"],
    "min_evidence_strength_ratio": 0.5,
    "expected_fallback_reasons": ["missing_data"],
    "min_source_refs": 1,
    "max_fallbacks": 8,
    "max_latency_ms": 5000
  }
}
```

Guidelines:

- Use `expected_entity_labels` whenever the scenario hinges on the v2 ontology — it's the difference between "the agent mentioned the topic" and "the agent surfaced an entity of the right type."
- Use `expected_fallback_reasons` for negative-space scenarios. The point is to *reward* honest gap declaration; without this assertion the agent would be penalized for the missing data and never positively credited for the right behavior.
- Use `forbidden_in_answer` instead of `must_not_contain` when the forbidden token is plausibly mentioned in metadata (file paths, source URIs). The narrower check only inspects the synthesized answer text.
- Use `min_evidence_strength_ratio` for "this question deserves authoritative data, not Slack chatter."
- For scenarios where you genuinely don't know the right answer, set `min_source_refs: 1` rather than asserting specific text — the agent should always cite something.

When `expected_entity_labels` doesn't fire as you expect, the response shape may carry records under an answer-level bucket key the evaluator doesn't yet recognize — see `_iter_records` in `evaluator.py`.

## Known Limitations

- **HTTP-E2E record paths**: The in-process router does not wire a live reconciliation agent, so `seed_record` returns skipped. Episodes and queries work fully.
- **API seed errors**: If the target pot does not contain the configured `repo_name`, record seeding returns `repo_not_in_pot`. This is expected and does not block the benchmark.
- **LLM variance**: API mode uses live LLM extraction; exact keyword matches (`must_contain`) may vary between runs. The dataset is tuned for stable extraction, but occasional variance is normal.
